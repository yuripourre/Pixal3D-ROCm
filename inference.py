import os
import argparse
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# expandable_segments reduces fragmentation on both CUDA and ROCm allocators
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"]  = "expandable_segments:True"
# Enable AOTriton-backed flash / memory-efficient attention on AMD gfx1xxx GPUs.
# Without this flag PyTorch SDPA silently falls back to the O(N²) math kernel.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

_TIMING = int(os.environ.get("PIXAL_TIMING", "0"))

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict, backbone=None, naf_model=None):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config, backbone=backbone)
    model.eval()
    if naf_model is not None and model.use_naf_upsample:
        # Inject shared NAF instance directly, skipping per-extractor _load_naf().
        model.naf_model = naf_model
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False, smart_vram=False):
    _t_init = time.perf_counter()

    # Load all 7 DiT/decoder checkpoints from disk.
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    if _TIMING:
        print(f"[TIMING]   model_load:dits={time.perf_counter()-_t_init:.1f}s", flush=True)

    # Load the frozen DINOv3 ViT-L backbone exactly ONCE and share it across
    # all 4 DinoV3ProjFeatureExtractor instances (was loaded 4x, each ~5-10 s).
    _t = time.perf_counter()
    from transformers import DINOv3ViTModel as _DINOv3ViTModel
    _dino_model_name = IMAGE_COND_CONFIGS["ss"]["model_name"]
    print(f"[ImageCond] Loading shared DINOv3 backbone ({_dino_model_name})...")
    _shared_backbone = _DINOv3ViTModel.from_pretrained(_dino_model_name)
    _shared_backbone.eval()
    _shared_backbone.requires_grad_(False)
    if _TIMING:
        print(f"[TIMING]   model_load:dino_backbone={time.perf_counter()-_t:.1f}s", flush=True)

    # Load NAF upsampler ONCE and share across the 3 NAF-using extractors.
    _t = time.perf_counter()
    import torch.hub as _hub
    print("[NAF] Loading shared NAF upsampler weights...")
    _shared_naf = _hub.load(
        "valeoai/NAF", "naf", pretrained=True, device="cpu", trust_repo=True
    )
    _shared_naf.eval()
    _shared_naf.requires_grad_(False)
    if _TIMING:
        print(f"[TIMING]   model_load:naf={time.perf_counter()-_t:.1f}s", flush=True)

    # Build the 4 extractors — construction now costs only ProjGrid creation,
    # not ViT-L or NAF loading.
    _t = time.perf_counter()
    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(
        IMAGE_COND_CONFIGS["ss"], backbone=_shared_backbone, naf_model=_shared_naf)
    pipeline.image_cond_model_shape_512 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_512"], backbone=_shared_backbone, naf_model=_shared_naf)
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["shape_1024"], backbone=_shared_backbone, naf_model=_shared_naf)
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(
        IMAGE_COND_CONFIGS["tex_1024"], backbone=_shared_backbone, naf_model=_shared_naf)
    if _TIMING:
        print(f"[TIMING]   model_load:extractors={time.perf_counter()-_t:.1f}s", flush=True)

    _cond_model_attrs = [
        'image_cond_model_ss', 'image_cond_model_shape_512',
        'image_cond_model_shape_1024', 'image_cond_model_tex_1024',
    ]
    # NAF is already pre-loaded and injected; no per-extractor _load_naf() needed.

    if smart_vram:
        # Smart-VRAM mode: small resident models (conditioning + decoders) live on
        # GPU permanently; large DiTs stream to GPU on demand then are freed with
        # del + empty_cache (no wasteful CPU copy-back for single-run workloads).
        pipeline.low_vram = True
        pipeline.smart_vram = True
        pipeline._device = torch.device(device)

        # Resident set: 4 conditioning models + 3 decoders + rembg (~4-5 GB total)
        for attr in _cond_model_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None:
                m.to(device)
        for name in ['sparse_structure_decoder', 'shape_slat_decoder', 'tex_slat_decoder']:
            if name in pipeline.models:
                pipeline.models[name].to(device)
        if pipeline.rembg_model is not None:
            pipeline.rembg_model.to(device)
        # Large DiTs stay on CPU: sparse_structure_flow_model,
        # shape_slat_flow_model_512/1024, tex_slat_flow_model_1024
        print("[Pipeline] Smart-VRAM mode: resident models on GPU, large DiTs stream on demand.")
    elif low_vram:
        # Low-VRAM mode: all models stay on CPU, loaded to GPU one at a time.
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once.
        pipeline.low_vram = False
        pipeline.cuda()
        for attr in _cond_model_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None:
                m.cuda()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    smart_vram: bool = False,
    resolution: int = -1,
):
    # Load models
    _t_total = time.perf_counter()
    _t = time.perf_counter()
    pipeline = init_pipeline(model_path, low_vram=low_vram, smart_vram=smart_vram)
    if _TIMING:
        print(f"[TIMING] model_load={time.perf_counter()-_t:.1f}s", flush=True)

    # Preprocess image first — rembg loads to GPU for this call, then offloads.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        _t = time.perf_counter()
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cuda")
        print("[Inference] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
        if _TIMING:
            print(f"[TIMING] camera_estimation={time.perf_counter()-_t:.1f}s", flush=True)
    os.remove(tmp_path)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if (low_vram or smart_vram) else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    _t = time.perf_counter()
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    if _TIMING:
        print(f"[TIMING] pipeline_run={time.perf_counter()-_t:.1f}s", flush=True)
    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    _t = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    if _TIMING:
        print(f"[TIMING] to_glb={time.perf_counter()-_t:.1f}s", flush=True)
        print(f"[TIMING] total={time.perf_counter()-_t_total:.1f}s", flush=True)

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--smart_vram", action="store_true",
                        help="Enable smart-VRAM mode: small conditioning models and decoders stay GPU-resident; "
                             "large DiTs stream to GPU on demand and are freed after use. "
                             "Faster than --low_vram with similar peak VRAM. Pins to 1024 resolution.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram/--smart_vram, else 1536.")

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        smart_vram=args.smart_vram,
        resolution=args.resolution,
    )
