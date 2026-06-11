from typing import *
import os
import warnings
import torch
from ..voxel import Voxel
import cumesh
from flex_gemm.ops.grid_sample import grid_sample_3d

DEFAULT_PYMESHLAB_MAX_HOLE_SIZE = 30
BOUNDING_BOX_SHRINK_THRESHOLD = 0.75


def _should_skip_fill_holes() -> bool:
    return os.environ.get("SKIP_FILL_HOLES", "").lower() in ("1", "true", "yes")


def _mesh_to_pymeshlab(vertices: torch.Tensor, faces: torch.Tensor):
    import pymeshlab as pml

    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(vertices.cpu().numpy(), faces.cpu().numpy()))
    return ms


def _read_pymeshlab_mesh(ms, device: torch.device):
    mesh = ms.current_mesh()
    vertices = torch.from_numpy(mesh.vertex_matrix()).float().to(device)
    faces = torch.from_numpy(mesh.face_matrix()).int().to(device)
    return vertices, faces


class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs
        
    @property
    def device(self):
        return self.vertices.device
        
    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )
        
    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)
        
    def cpu(self):
        return self.to('cpu')
    
    def _fill_holes_pymeshlab(self, max_hole_size: int = DEFAULT_PYMESHLAB_MAX_HOLE_SIZE) -> None:
        try:
            ms = _mesh_to_pymeshlab(self.vertices, self.faces)
            ms.meshing_close_holes(maxholesize=max_hole_size)
            new_vertices, new_faces = _read_pymeshlab_mesh(ms, self.device)
            self.vertices = new_vertices
            self.faces = new_faces
        except Exception as e:
            warnings.warn(
                f"fill_holes pymeshlab fallback failed ({e}); skipping.",
                RuntimeWarning,
                stacklevel=3,
            )

    def fill_holes(self, max_hole_perimeter=3e-2):
        if _should_skip_fill_holes():
            return

        vertices = self.vertices.clone().cuda().contiguous()
        faces = self.faces.clone().cuda().contiguous()

        try:
            mesh = cumesh.CuMesh()
            mesh.init(vertices, faces)
            mesh.get_edges()
            mesh.get_boundary_info()
            if mesh.num_boundaries == 0:
                return
            mesh.get_vertex_edge_adjacency()
            mesh.get_vertex_boundary_adjacency()
            mesh.get_manifold_boundary_adjacency()
            mesh.read_manifold_boundary_adjacency()
            mesh.get_boundary_connected_components()
            mesh.get_boundary_loops()
            if mesh.num_boundary_loops == 0:
                return
            mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
            new_vertices, new_faces = mesh.read()

            # Sanity check: CuMesh on HIP sometimes returns a corrupted mesh
            # (e.g., all vertices clipped to one half-space).  If the bounding
            # box shrank significantly in any axis, discard the result.
            orig_extent = vertices.max(dim=0).values - vertices.min(dim=0).values
            new_extent = new_vertices.max(dim=0).values - new_vertices.min(dim=0).values
            if (new_extent < orig_extent * BOUNDING_BOX_SHRINK_THRESHOLD).any():
                warnings.warn(
                    "fill_holes returned a mesh with a significantly smaller bounding box "
                    f"(orig {orig_extent.tolist()}, new {new_extent.tolist()}); "
                    "discarding fill_holes result.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fill_holes_pymeshlab()
                return

            self.vertices = new_vertices.to(self.device)
            self.faces = new_faces.to(self.device)
        except Exception as e:
            warnings.warn(
                f"fill_holes failed ({e}); trying pymeshlab fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fill_holes_pymeshlab()

    def smooth(self, iterations: int = 10) -> None:
        if iterations <= 0:
            return

        try:
            ms = _mesh_to_pymeshlab(self.vertices, self.faces)
            ms.apply_coord_taubin_smoothing(stepsmoothnum=iterations)
            new_vertices, new_faces = _read_pymeshlab_mesh(ms, self.device)
            self.vertices = new_vertices
            self.faces = new_faces
        except Exception as e:
            warnings.warn(
                f"smooth failed ({e}); skipping.",
                RuntimeWarning,
                stacklevel=2,
            )

    def repair_topology(self) -> None:
        try:
            ms = _mesh_to_pymeshlab(self.vertices, self.faces)
            ms.meshing_repair_non_manifold_edges(method=0)
            ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
            new_vertices, new_faces = _read_pymeshlab_mesh(ms, self.device)
            self.vertices = new_vertices
            self.faces = new_faces
        except Exception as e:
            warnings.warn(
                f"repair_topology failed ({e}); skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
        
    def remove_faces(self, face_mask: torch.Tensor):
        vertices = self.vertices.clone().cuda().contiguous()
        faces = self.faces.clone().cuda().contiguous()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_faces(face_mask)
        new_vertices, new_faces = mesh.read()
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)
        
    def simplify(self, target=1000000, verbose: bool=False, options: dict={}):
        vertices = self.vertices.clone().cuda().contiguous()
        faces = self.faces.clone().cuda().contiguous()
        
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()
        
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)


class TextureFilterMode:
    CLOSEST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


class AlphaMode:
    OPAQUE = 0
    MASK = 1
    BLEND = 2


class Texture:
    def __init__(
        self,
        image: torch.Tensor,
        filter_mode: TextureFilterMode = TextureFilterMode.LINEAR,
        wrap_mode: TextureWrapMode = TextureWrapMode.REPEAT
    ):
        self.image = image
        self.filter_mode = filter_mode
        self.wrap_mode = wrap_mode

    def to(self, device, non_blocking=False):
        return Texture(
            self.image.to(device, non_blocking=non_blocking),
            self.filter_mode,
            self.wrap_mode,
        )


class PbrMaterial:
    def __init__(
        self,
        base_color_texture: Optional[Texture] = None,
        base_color_factor: Union[torch.Tensor, List[float]] = [1.0, 1.0, 1.0],
        metallic_texture: Optional[Texture] = None,
        metallic_factor: float = 1.0,
        roughness_texture: Optional[Texture] = None,
        roughness_factor: float = 1.0,
        alpha_texture: Optional[Texture] = None,
        alpha_factor: float = 1.0,
        alpha_mode: AlphaMode = AlphaMode.OPAQUE,
        alpha_cutoff: float = 0.5,
    ):
        self.base_color_texture = base_color_texture
        self.base_color_factor = torch.tensor(base_color_factor, dtype=torch.float32)[:3]
        self.metallic_texture = metallic_texture
        self.metallic_factor = metallic_factor
        self.roughness_texture = roughness_texture
        self.roughness_factor = roughness_factor
        self.alpha_texture = alpha_texture
        self.alpha_factor = alpha_factor
        self.alpha_mode = alpha_mode
        self.alpha_cutoff = alpha_cutoff

    def to(self, device, non_blocking=False):
        return PbrMaterial(
            base_color_texture=self.base_color_texture.to(device, non_blocking=non_blocking) if self.base_color_texture is not None else None,
            base_color_factor=self.base_color_factor.to(device, non_blocking=non_blocking),
            metallic_texture=self.metallic_texture.to(device, non_blocking=non_blocking) if self.metallic_texture is not None else None,
            metallic_factor=self.metallic_factor,
            roughness_texture=self.roughness_texture.to(device, non_blocking=non_blocking) if self.roughness_texture is not None else None,
            roughness_factor=self.roughness_factor,
            alpha_texture=self.alpha_texture.to(device, non_blocking=non_blocking) if self.alpha_texture is not None else None,
            alpha_factor=self.alpha_factor,
            alpha_mode=self.alpha_mode,
            alpha_cutoff=self.alpha_cutoff,
        )


class MeshWithPbrMaterial(Mesh):
    def __init__(self,
        vertices,
        faces,
        material_ids,
        uv_coords,
        materials: List[PbrMaterial],
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.material_ids = material_ids    # [M]
        self.uv_coords = uv_coords          # [M, 3, 2]
        self.materials = materials
        self.layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }

    def to(self, device, non_blocking=False):
        return MeshWithPbrMaterial(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.material_ids.to(device, non_blocking=non_blocking),
            self.uv_coords.to(device, non_blocking=non_blocking),
            [material.to(device, non_blocking=non_blocking) for material in self.materials],
        )


class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )
        
    def query_attrs(self, xyz):
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape,
            grid,
            mode='trilinear'
        )[0]
        return vertex_attrs
        
    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)
