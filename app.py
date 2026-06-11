import os
import subprocess
import argparse
import math
import time
import shutil
import socket
import cv2
import torch
import numpy as np
import base64
import io
import json
from datetime import datetime
from typing import *
from PIL import Image

import threading
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Lock for model initialization
init_lock = threading.Lock()

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

try:
    import spaces
except ImportError:
    class _SpacesStub:
        class GPU:
            def __init__(self, **kwargs):
                pass

            def __call__(self, fn):
                return fn

    spaces = _SpacesStub()

from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def _patch_gradio_lan_file_handling():
    """Gradio SSRF protection rejects private/LAN hostnames in file URLs.

    When clients connect via http://192.168.x.x, uploaded files are referenced
    by URLs on that host and async_ssrf_protected_download fails.  For local
    networks, resolve gradio file= paths on disk or fetch without SSRF checks.
    """
    import asyncio
    import ipaddress
    from urllib.parse import unquote, urlparse

    import httpx
    from gradio import processing_utils as pu

    _orig_download = pu.async_ssrf_protected_download
    _orig_get = pu.async_ssrf_protected_get

    def _is_private_host(host: str | None) -> bool:
        if not host or host == "localhost":
            return bool(host)
        try:
            ip = ipaddress.ip_address(host)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False

    def _path_from_gradio_file_url(url: str) -> str | None:
        if "file=" not in url:
            return None
        local_path = unquote(url.split("file=", 1)[1].split("?")[0])
        if os.path.isfile(local_path):
            return local_path
        return None

    async def _lan_async_ssrf_protected_download(url: str, cache_dir: str) -> str:
        local_path = _path_from_gradio_file_url(url)
        if local_path:
            return pu.save_file_to_cache(local_path, cache_dir)

        if _is_private_host(urlparse(url).hostname):
            return await asyncio.to_thread(pu.unsafe_download, url, cache_dir)

        return await _orig_download(url, cache_dir)

    async def _lan_async_ssrf_protected_get(url: str) -> httpx.Response:
        if _is_private_host(urlparse(url).hostname):
            async with httpx.AsyncClient() as client:
                return await client.get(url, follow_redirects=False)
        return await _orig_get(url)

    pu.async_ssrf_protected_download = _lan_async_ssrf_protected_download
    pu.async_ssrf_protected_get = _lan_async_ssrf_protected_get


_patch_gradio_lan_file_handling()

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8

# Cascade parameters
CASCADE_LR_RESOLUTION = 512
CASCADE_MAX_NUM_TOKENS = 49152

# MoGe defaults
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

# Image Cond Model configs
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
        model.naf_model = naf_model
    return model

def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model

# Global instances (lazy loaded or loaded at start)
pipeline = None
moge_model = None
envmap = None
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"
SMART_VRAM = os.environ.get("SMART_VRAM", "0") == "1"


def _vram_efficient_mode() -> bool:
    """True when models are not all permanently resident on GPU."""
    return LOW_VRAM or SMART_VRAM


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def init_models():
    global pipeline, moge_model, envmap
    with init_lock:
        if pipeline is not None:
            return

        # GPU / CUDA Diagnostics (runs when GPU is allocated)
        import subprocess as _sp
        print("=" * 60)
        print("[Diagnostics] PyTorch version:", torch.__version__)
        print("[Diagnostics] CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[Diagnostics] CUDA version:", torch.version.cuda)
            print("[Diagnostics] cuDNN version:", torch.backends.cudnn.version())
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"[Diagnostics] GPU {i}: {name}, sm_{cap[0]}{cap[1]}, {mem:.1f} GB")
        try:
            res = _sp.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            print("[Diagnostics] nvidia-smi:", res.stdout.strip())
        except Exception as e:
            print(f"[Diagnostics] nvidia-smi failed: {e}")
        print("=" * 60)

        model_path = "TencentARC/Pixal3D"
        print(f"[Pipeline] Loading from {model_path}...")
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

        # Load shared backbone + NAF once instead of 4x and 3x respectively.
        from transformers import DINOv3ViTModel as _DINOv3ViTModel
        _dino_model_name = IMAGE_COND_CONFIGS["ss"]["model_name"]
        print(f"[ImageCond] Loading shared DINOv3 backbone ({_dino_model_name})...")
        _shared_backbone = _DINOv3ViTModel.from_pretrained(_dino_model_name)
        _shared_backbone.eval()
        _shared_backbone.requires_grad_(False)

        import torch.hub as _hub
        print("[NAF] Loading shared NAF upsampler weights...")
        _shared_naf = _hub.load(
            "valeoai/NAF", "naf", pretrained=True, device="cpu", trust_repo=True
        )
        _shared_naf.eval()
        _shared_naf.requires_grad_(False)

        print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
        pipeline.image_cond_model_ss = build_image_cond_model(
            IMAGE_COND_CONFIGS["ss"], backbone=_shared_backbone, naf_model=_shared_naf)
        pipeline.image_cond_model_shape_512 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_512"], backbone=_shared_backbone, naf_model=_shared_naf)
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_1024"], backbone=_shared_backbone, naf_model=_shared_naf)
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["tex_1024"], backbone=_shared_backbone, naf_model=_shared_naf)

        _cond_model_attrs = [
            'image_cond_model_ss', 'image_cond_model_shape_512',
            'image_cond_model_shape_1024', 'image_cond_model_tex_1024',
        ]
        if SMART_VRAM:
            pipeline.low_vram = True
            pipeline.smart_vram = True
            pipeline._device = torch.device("cuda")
            for attr in _cond_model_attrs:
                m = getattr(pipeline, attr, None)
                if m is not None:
                    m.cuda()
            for name in ['sparse_structure_decoder', 'shape_slat_decoder', 'tex_slat_decoder']:
                if name in pipeline.models:
                    pipeline.models[name].cuda()
            if pipeline.rembg_model is not None:
                pipeline.rembg_model.cuda()
            _shared_naf.cuda()
            print("[Pipeline] Smart-VRAM mode: resident models on GPU, large DiTs stream on demand.")
        elif LOW_VRAM:
            pipeline._device = torch.device("cuda")
            pipeline.low_vram = True
            print("[Pipeline] Low-VRAM mode enabled.")
        else:
            pipeline.low_vram = False
            pipeline.cuda()
            pipeline.image_cond_model_ss.cuda()
            pipeline.image_cond_model_shape_512.cuda()
            pipeline.image_cond_model_shape_1024.cuda()
            pipeline.image_cond_model_tex_1024.cuda()
            _shared_naf.cuda()

        print("[MoGe-2] Loading model for camera estimation...")
        if _vram_efficient_mode():
            moge_model = load_moge_model(device="cpu")
            print("[MoGe-2] VRAM-efficient mode: MoGe stays on CPU, loaded to GPU on-demand.")
        else:
            moge_model = load_moge_model(device="cuda")

        print("[EnvMap] Loading environment maps...")
        _base = os.path.dirname(os.path.abspath(__file__))
        _envmap_device = 'cpu' if _vram_efficient_mode() else 'cuda'
        envmap = {
            'forest': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'sunset': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/sunset.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'courtyard': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
        }

# ============================================================================
# Utilities
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

def get_camera_params_wild_moge(image_path, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    if _vram_efficient_mode():
        moge_model.to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    if _vram_efficient_mode():
        moge_model.cpu()
        torch.cuda.empty_cache()
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

def pack_state(shape_slat, tex_slat, res):
    state_data = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    import random
    state_path = os.path.join(TMP_DIR, f"state_{int(time.time()*1000)}_{random.randint(0,9999):04d}.npz")
    np.savez_compressed(state_path, **state_data)
    return state_path

def unpack_state(state_path):
    data = np.load(state_path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])

# ============================================================================
# Progress Tracking (file-based, cross-process safe for @spaces.GPU)
# ============================================================================

import asyncio
from fastapi.responses import JSONResponse
from fastapi import Request

PROGRESS_DIR = os.path.join(TMP_DIR, '_progress')
os.makedirs(PROGRESS_DIR, exist_ok=True)

_thread_local = threading.local()

def _progress_file(session_id: str) -> str:
    """Return path to a session's progress JSON file."""
    return os.path.join(PROGRESS_DIR, f"{session_id}.json")

def _reset_progress(session_id: str):
    _thread_local.active_session = session_id
    _write_progress_file(session_id, {"stage": "Initializing...", "step": 0, "total": 0, "done": False})

def _update_progress(stage: str, step: int, total: int):
    session_id = getattr(_thread_local, 'active_session', '')
    if session_id:
        _write_progress_file(session_id, {"stage": stage, "step": step, "total": total, "done": False})

def _finish_progress():
    session_id = getattr(_thread_local, 'active_session', '')
    if session_id:
        _write_progress_file(session_id, {"done": True})

def _write_progress_file(session_id: str, data: dict):
    """Atomically write progress JSON to a file (cross-process safe)."""
    path = _progress_file(session_id)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        pass

# Monkey-patch tqdm to intercept progress
import tqdm as _tqdm_module

_original_tqdm = _tqdm_module.tqdm

class _TqdmProgressInterceptor(_original_tqdm):
    """Wraps tqdm to push progress updates to SSE."""
    def __init__(self, *args, **kwargs):
        self._stage_desc = kwargs.get('desc', 'Processing')
        super().__init__(*args, **kwargs)
    
    def set_description(self, desc=None, refresh=True):
        self._stage_desc = desc or 'Processing'
        super().set_description(desc, refresh)
    
    def update(self, n=1):
        super().update(n)
        _update_progress(self._stage_desc, self.n, self.total or 0)

# Patch tqdm globally
_tqdm_module.tqdm = _TqdmProgressInterceptor
# Also patch the direct import in the sampler module and render_utils
import pixal3d.pipelines.samplers.flow_euler as _fe_module
_fe_module.tqdm = _TqdmProgressInterceptor
import pixal3d.utils.render_utils as _ru_module
_ru_module.tqdm = _TqdmProgressInterceptor
import o_voxel.postprocess as _ovp_module
_ovp_module.tqdm = _TqdmProgressInterceptor

# ============================================================================
# API Implementation
# ============================================================================

app = Server()

@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/app_config")
async def get_config():
    """Return server configuration for frontend (e.g. LOW_VRAM mode)."""
    return JSONResponse({"low_vram": LOW_VRAM, "smart_vram": SMART_VRAM})

@app.get("/progress")
async def progress_poll(request: Request):
    """Polling endpoint for real-time progress updates during generation."""
    session_id = request.query_params.get("session_id", "")
    path = _progress_file(session_id)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return JSONResponse(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"stage": "Waiting...", "step": 0, "total": 0, "done": False})

@app.api()
@spaces.GPU(duration=30)
def preprocess(image: FileData) -> FileData:
    init_models()
    img = Image.open(image["path"])
    processed = pipeline.preprocess_image(img)
    out_path = os.path.join(TMP_DIR, f"preprocessed_{int(time.time()*1000)}.png")
    processed.save(out_path)
    return FileData(path=out_path)

@app.api()
@spaces.GPU(duration=120)
def generate_3d(
    image: FileData, 
    seed: int, 
    resolution: int,
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
    manual_fov: float = -1.0,
    fov_unit: str = "deg",
    session_id: str = "",
) -> Dict:
    init_models()
    _reset_progress(session_id)
    _update_progress("Preprocessing & Camera Estimation", 0, 1)
    
    torch.manual_seed(seed)
    hr_resolution = int(resolution)
    
    img = Image.open(image["path"])
    # Image is already preprocessed by /preprocess endpoint, use directly
    image_preprocessed = img
    temp_processed_path = os.path.join(TMP_DIR, f"temp_proc_{session_id[:8]}_{int(time.time()*1000)}.png")
    image_preprocessed.save(temp_processed_path)
    
    if manual_fov > 0:
        # Convert to radians based on unit
        if fov_unit == "rad":
            camera_angle_x = float(manual_fov)
            fov_deg = math.degrees(manual_fov)
        else:
            camera_angle_x = math.radians(manual_fov)
            fov_deg = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - WILD_EXTEND_PIXEL, WILD_IMAGE_RESOLUTION - 1 + WILD_EXTEND_PIXEL]),
            WILD_MESH_SCALE, WILD_IMAGE_RESOLUTION
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': WILD_MESH_SCALE}
        print(f"[Camera] Using manual FOV: {fov_deg:.2f}° ({camera_angle_x:.4f} rad), distance: {distance:.4f}")
    else:
        camera_params = get_camera_params_wild_moge(
            temp_processed_path, device="cuda",
            mesh_scale=WILD_MESH_SCALE, extend_pixel=WILD_EXTEND_PIXEL,
            image_resolution=WILD_IMAGE_RESOLUTION,
        )
    _update_progress("Preprocessing & Camera Estimation", 1, 1)
    
    ss_sampler_override = {"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
                           "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t}
    shape_sampler_override = {"steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
                              "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t}
    tex_sampler_override = {"steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
                            "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t}

    pipeline_type = f"{hr_resolution}_cascade"
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
        max_num_tokens=CASCADE_MAX_NUM_TOKENS,
    )
    
    mesh = mesh_list[0]
    state_path = pack_state(shape_slat, tex_slat, res)
    
    render_files = {}
    preview_glb_path = None
    try:
        from pixal3d.utils.fallback_mesh_renderer import export_preview_glb, nvdiffrast_available
        use_nvdiffrast = nvdiffrast_available()
        if not use_nvdiffrast:
            print("[Render] nvdiffrast not available — exporting vertex-colored preview GLB.")
            _update_progress("Rendering views", 0, 1)
            preview_glb_path = os.path.abspath(
                os.path.join(TMP_DIR, f"preview_{int(time.time()*1000)}.glb")
            )
            export_preview_glb(mesh, preview_glb_path)
            _update_progress("Rendering views", 1, 1)
        else:
            _update_progress("Rendering views", 0, STEPS)
            mesh.simplify(16777216)
            cam_dist = camera_params['distance']
            near = max(0.01, cam_dist - 2.0)
            far = cam_dist + 10.0
            if _vram_efficient_mode():
                for v in envmap.values():
                    v.image = v.image.cuda()
                    if hasattr(v, '_nvdiffrec_envlight'):
                        del v._nvdiffrec_envlight

            preview_resolution = 1024
            renders = render_utils.render_proj_aligned_video(
                mesh, camera_angle_x=camera_params['camera_angle_x'],
                distance=cam_dist, resolution=preview_resolution,
                num_frames=STEPS, envmap=envmap,
                near=near, far=far,
            )
            if _vram_efficient_mode():
                for v in envmap.values():
                    if hasattr(v, '_nvdiffrec_envlight'):
                        del v._nvdiffrec_envlight
                    v.image = v.image.cpu()
                torch.cuda.empty_cache()
            _update_progress("Rendering views", STEPS, STEPS)

            for mode_key, frames in renders.items():
                mode_files = []
                for i, frame in enumerate(frames):
                    p = os.path.abspath(os.path.join(TMP_DIR, f"render_{mode_key}_{i}_{int(time.time()*1000)}.jpg"))
                    Image.fromarray(frame).save(p, quality=85)
                    mode_files.append(FileData(path=p))
                render_files[mode_key] = mode_files
    except Exception as e:
        print(f"[Render] Preview render failed ({e}) — skipping preview renders.")

    _finish_progress()
    return {
        "render_paths": render_files,
        "preview_glb": preview_glb_path,
        "state_path": os.path.abspath(state_path),
        "camera_angle_x": camera_params['camera_angle_x'],
        "distance": camera_params['distance'],
    }

@app.api()
@spaces.GPU(duration=240)
def extract_glb_api(
    state_path: str,
    decimation_target: int,
    texture_size: int,
    dc_resolution: int = 256,
    smooth_iterations: int = 0,
    fill_holes: bool = True,
    session_id: str = "",
) -> FileData:
    init_models()
    _reset_progress(session_id)
    _update_progress("Decoding latent", 0, 1)

    if fill_holes:
        os.environ.pop("SKIP_FILL_HOLES", None)
    else:
        os.environ["SKIP_FILL_HOLES"] = "1"

    shape_slat, tex_slat, res = unpack_state(state_path)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    _update_progress("Decoding latent", 1, 1)

    if smooth_iterations > 0:
        mesh.smooth(smooth_iterations)

    os.environ["DC_RESOLUTION"] = str(dc_resolution)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    
    out_glb = os.path.join(TMP_DIR, f"result_{int(time.time()*1000)}.glb")
    glb.export(out_glb, extension_webp=True)
    _finish_progress()
    return FileData(path=out_glb)

# Mount assets and tmp for direct access
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Pixal3D Demo Server")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models lazy-load to GPU per stage.")
    parser.add_argument("--smart_vram", action="store_true",
                        help="Enable smart-VRAM mode: conditioning + decoders stay on GPU; "
                             "large DiTs stream on demand. Faster than --low_vram.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default 0.0.0.0 for LAN access).")
    parser.add_argument("--port", type=int, default=7861,
                        help="Port to listen on (default 7861).")
    parser.add_argument("--share", action="store_true",
                        help="Create a temporary public Gradio tunnel (off by default).")
    parser.add_argument("--skip-utils3d-reinstall", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Skip forced utils3d wheel reinstall on startup (default: skip).")
    args, remaining = parser.parse_known_args()
    if args.low_vram:
        LOW_VRAM = True
    if args.smart_vram:
        SMART_VRAM = True

    if not args.skip_utils3d_reinstall:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
            "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"
        ], check=True)

    import socket
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        _sock.bind((args.host, args.port))
    except OSError as e:
        _sock.close()
        raise SystemExit(
            f"[Server] Port {args.port} is already in use on {args.host}. "
            f"Stop the other process or choose another port, e.g. PORT={args.port + 1} ./start.sh"
        ) from e
    _sock.close()

    init_models()

    lan_ip = _local_ip()
    print(f"[Server] Local:   http://127.0.0.1:{args.port}")
    print(f"[Server] LAN:     http://{lan_ip}:{args.port}")
    if args.share:
        print("[Server] Public Gradio tunnel enabled via --share")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )