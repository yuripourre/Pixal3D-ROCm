"""
Software mesh preview renderer for platforms without nvdiffrast (e.g. AMD ROCm).

Produces the same output dict shape as render_utils.render_frames for UI previews.
Uses face subsampling (not CuMesh simplify) so multi-million-face meshes stay fast.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..renderers.mesh_renderer import intrinsics_to_projection
from ..representations import MeshWithVoxel

FALLBACK_PREVIEW_FACES = 4_000
FALLBACK_PREVIEW_RES = 512
PREVIEW_GLB_FACES = 50_000

# Same axis rotation as full GLB export in app.py / inference.py
GLB_EXPORT_ROTATION = np.array([
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def nvdiffrast_available() -> bool:
    try:
        import nvdiffrast  # noqa: F401
        return True
    except ImportError:
        return False


def _subsample_faces(faces: torch.Tensor, max_faces: int) -> torch.Tensor:
    n_faces = faces.shape[0]
    if n_faces <= max_faces:
        return faces
    idx = torch.linspace(0, n_faces - 1, max_faces, device=faces.device).long()
    return faces[idx]


def _compact_mesh(verts: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only vertices referenced by faces (avoids projecting millions of verts)."""
    used = torch.unique(faces.reshape(-1))
    compact_verts = verts[used]
    remap = torch.full((verts.shape[0],), -1, dtype=torch.long, device=verts.device)
    remap[used] = torch.arange(used.shape[0], device=verts.device)
    return compact_verts, remap[faces.long()]


def _face_geometry(
    verts: torch.Tensor,
    mesh: MeshWithVoxel,
    faces: torch.Tensor,
):
    f = faces.long()
    v0 = verts[f[:, 0]]
    v1 = verts[f[:, 1]]
    v2 = verts[f[:, 2]]
    face_normal = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normal = torch.nn.functional.normalize(face_normal, dim=1)
    centers = (v0 + v1 + v2) / 3.0

    center_attrs = mesh.query_attrs(centers)
    base_color = center_attrs[:, mesh.layout["base_color"]].clamp(0.0, 1.0)

    return face_normal, centers, base_color


def _rotation_y(angle: float, device: torch.device) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.float32,
        device=device,
    )


def _project_vertices(
    vertices: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: float,
    far: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = vertices.device
    verts_homo = torch.cat(
        [vertices, torch.ones(vertices.shape[0], 1, device=device)], dim=1
    )
    perspective = intrinsics_to_projection(intrinsics, near, far)
    full_proj = perspective @ extrinsics
    clip = (verts_homo @ full_proj.T).cpu().numpy()
    w = np.clip(clip[:, 3:4], 1e-8, None)
    ndc = clip[:, :3] / w
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * resolution
    # nvdiffrast uses y-down NDC (y=-1 at top); no extra Y flip needed here.
    screen_y = (ndc[:, 1] * 0.5 + 0.5) * resolution
    depth = ndc[:, 2]
    return np.stack([screen_x, screen_y], axis=1), depth


def _raster_order(
    verts: torch.Tensor,
    faces: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray]:
    screen_xy, depth = _project_vertices(
        verts, extrinsics, intrinsics, near, far, resolution
    )
    face_idx = faces.cpu().numpy()
    tri_xy = screen_xy[face_idx]
    tri_depth = depth[face_idx].mean(axis=1)
    order = np.argsort(tri_depth)[::-1]
    return tri_xy, order


def _sample_latlong(env_image: torch.Tensor, directions: np.ndarray) -> np.ndarray:
    env = env_image.detach().float().cpu().numpy()
    h, w = env.shape[:2]
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    u = np.arctan2(x, -z) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(y, -1.0, 1.0)) / np.pi
    px = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    py = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    return env[py, px]


def _rasterize_ordered(
    tri_xy: np.ndarray,
    order: np.ndarray,
    face_colors: np.ndarray,
    resolution: int,
    bg_color: tuple[float, float, float],
) -> np.ndarray:
    bg = tuple(int(c * 255) for c in bg_color)
    image = Image.new("RGB", (resolution, resolution), bg)
    draw = ImageDraw.Draw(image)
    for idx in order:
        pts = [(float(x), float(y)) for x, y in tri_xy[idx]]
        if not all(np.isfinite(x) and np.isfinite(y) for x, y in pts):
            continue
        color = tuple(int(c) for c in np.clip(face_colors[idx] * 255.0, 0, 255))
        draw.polygon(pts, fill=color)
    return np.array(image, dtype=np.uint8)


def _camera_basis(extrinsics: torch.Tensor) -> np.ndarray:
    extr = extrinsics.detach().cpu().numpy()
    return -extr[:3, :3].T @ extr[:3, 3]


def _render_modes_for_frame(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normal: torch.Tensor,
    centers: torch.Tensor,
    face_base_color: torch.Tensor,
    angle: float,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int,
    near: float,
    far: float,
    bg_color: tuple[float, float, float],
    envmap: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, np.ndarray]:
    rot = _rotation_y(angle, verts.device)
    rotated_verts = verts @ rot.T
    rotated_normals = face_normal @ rot.T
    rotated_centers = centers @ rot.T

    tri_xy, order = _raster_order(
        rotated_verts, faces, extrinsics, intrinsics, resolution, near, far
    )

    cam_pos = _camera_basis(extrinsics)
    view_dir = rotated_centers.detach().cpu().numpy() - cam_pos
    view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)
    normals_np = rotated_normals.detach().cpu().numpy()
    ndotl = np.clip(np.sum(normals_np * (-view_dir), axis=1), 0.0, 1.0)

    normal_rgb = (normals_np * 0.5 + 0.5).clip(0.0, 1.0)
    clay_rgb = np.stack([(1.0 - ndotl).clip(0.0, 1.0)] * 3, axis=1)
    base_rgb = face_base_color.detach().cpu().numpy()

    out = {
        "normal": _rasterize_ordered(tri_xy, order, normal_rgb, resolution, bg_color),
        "clay": _rasterize_ordered(tri_xy, order, clay_rgb, resolution, bg_color),
        "base_color": _rasterize_ordered(tri_xy, order, base_rgb, resolution, bg_color),
    }

    if envmap:
        reflect = view_dir - 2.0 * np.sum(normals_np * view_dir, axis=1, keepdims=True) * normals_np
        reflect = reflect / (np.linalg.norm(reflect, axis=1, keepdims=True) + 1e-8)
        for key, env_image in envmap.items():
            env_rgb = _sample_latlong(env_image, reflect)
            shaded = (base_rgb * (0.25 + 0.75 * ndotl[:, None]) + env_rgb * 0.35).clip(0.0, 1.0)
            shaded_key = f"shaded_{key}" if key else "shaded"
            out[shaded_key] = _rasterize_ordered(tri_xy, order, shaded, resolution, bg_color)
    return out


def render_proj_aligned_video_fallback(
    sample: MeshWithVoxel,
    camera_angle_x: float,
    distance: float,
    resolution: int = FALLBACK_PREVIEW_RES,
    num_frames: int = 8,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    envmap: Optional[Union[dict, object]] = None,
    near: float = 0.01,
    far: float = 100.0,
    max_faces: int = FALLBACK_PREVIEW_FACES,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, List[np.ndarray]]:
    from .render_utils import proj_camera_to_render_params

    faces = _subsample_faces(sample.faces, max_faces)
    verts, faces = _compact_mesh(sample.vertices, faces)
    face_normal, centers, face_base_color = _face_geometry(verts, sample, faces)
    print(
        f"[Render] Software preview: {faces.shape[0]} faces, "
        f"{resolution}px, {num_frames} frames",
        flush=True,
    )

    extr_first, intr_first = proj_camera_to_render_params(camera_angle_x, distance)

    env_images: Optional[Dict[str, torch.Tensor]] = None
    if envmap is not None:
        env_images = {}
        for key, value in envmap.items():
            image = value.image if hasattr(value, "image") else value
            env_images[key] = image

    angles = torch.linspace(0, 2 * math.pi, num_frames + 1)[:num_frames]
    rets: Dict[str, List[np.ndarray]] = {}
    for frame_idx, angle in enumerate(angles):
        frame = _render_modes_for_frame(
            verts,
            faces,
            face_normal,
            centers,
            face_base_color,
            float(angle),
            extr_first,
            intr_first,
            resolution,
            near,
            far,
            bg_color,
            env_images,
        )
        for key, image in frame.items():
            rets.setdefault(key, []).append(image)
        if progress_callback is not None:
            progress_callback(frame_idx + 1, num_frames)

    return rets


def export_preview_glb(
    sample: MeshWithVoxel,
    out_path: str,
    max_faces: int = PREVIEW_GLB_FACES,
) -> None:
    """Export a vertex-colored GLB for interactive preview (no UV baking)."""
    import trimesh

    faces = _subsample_faces(sample.faces, max_faces)
    verts, faces = _compact_mesh(sample.vertices, faces)
    colors = sample.query_attrs(verts)[:, sample.layout["base_color"]]
    colors_u8 = (colors.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    tm = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=colors_u8,
        process=False,
    )
    tm.apply_transform(GLB_EXPORT_ROTATION)
    tm.export(out_path)
    print(
        f"[Render] Preview GLB: {faces.shape[0]} faces -> {out_path}",
        flush=True,
    )
