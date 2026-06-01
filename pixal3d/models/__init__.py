import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _load_safetensors(filename: str, device: str = "cpu") -> dict:
    """
    Load a safetensors file, with fallback handling for non-standard dtypes
    such as C64/C128 complex tensors that the standard safetensors Rust
    backend rejects.

    Load strategy (fastest to slowest):
    1. safetensors.safe_open  — Rust mmap-backed, single-copy, handles the
       checkpoints correctly even when load_file fails on some ROCm builds.
    2. Python memoryview fallback — single-copy: zero-copy memoryview slice
       into the mmap, then one .clone() to own the data.  Needed when the Rust
       backend does not support a dtype (C64/C128) or raises any exception.
    """
    import struct
    import json
    import mmap
    import torch

    DTYPE_MAP = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    COMPLEX_REAL_MAP = {
        "C64": torch.float32,
        "C128": torch.float64,
    }

    # Fast path: safe_open uses the Rust mmap backend and avoids ROCm issues
    # that can occur with safetensors.torch.load_file on some builds.
    try:
        from safetensors import safe_open
        tensors = {}
        with safe_open(filename, framework="pt", device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    except Exception:
        pass

    # Python fallback: single-copy via memoryview (zero-copy view into the mmap,
    # then one .clone() to produce an owned tensor before closing the mmap).
    tensors = {}
    with open(filename, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm)
        data_offset = 8 + header_len

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            abs_start = data_offset + start
            abs_end = data_offset + end

            # Use a named slice so we can explicitly delete it after cloning,
            # which releases the exported-pointer reference on the mmap.
            chunk = mv[abs_start:abs_end]
            if dtype_str in COMPLEX_REAL_MAP:
                real_dtype = COMPLEX_REAL_MAP[dtype_str]
                _tmp = torch.frombuffer(chunk, dtype=real_dtype)
                t = torch.view_as_complex(_tmp.reshape(shape + [2])).clone()
                del _tmp
            elif dtype_str in DTYPE_MAP:
                _tmp = torch.frombuffer(chunk, dtype=DTYPE_MAP[dtype_str])
                t = _tmp.reshape(shape).clone()
                del _tmp
            else:
                raise ValueError(f"Unknown dtype {dtype_str!r} for tensor {key!r}")
            del chunk  # release exported pointer on mv/mm

            tensors[key] = t.to(device)

        # All chunk/tmp references are gone; release mv then close the mmap.
        del mv
        mm.close()

    return tensors


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.

    Environment variables:
        FAST_INIT  (default "1") When enabled, the model is constructed on the
                   meta device to skip the expensive random weight
                   initialisation (~7s per 1.3B-param DiT) that is immediately
                   overwritten by the checkpoint anyway, then materialised and
                   filled via copy_ (which preserves the model's parameter dtype,
                   so the result is bit-for-bit identical to a normal load).  The
                   fast path is used only when every parameter and buffer is
                   present in the checkpoint (no missing keys would otherwise be
                   left uninitialised); any mismatch falls back to the standard
                   construct-then-load path.  Set FAST_INIT=0 to disable.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)

    state = _load_safetensors(model_file)

    model = None
    if os.environ.get("FAST_INIT", "1") == "1":
        model = _try_fast_init(config, state, **kwargs)

    if model is None:
        # Standard path: construct with full weight init, then overwrite via copy_.
        model = __getattr__(config['name'])(**config['args'], **kwargs)
        model.load_state_dict(state, strict=False)

    return model


# In-place weight initialisers that perform expensive RNG fills.  During a fast
# load every parameter is overwritten by the checkpoint, so these fills are pure
# waste (~7s per 1.3B-param DiT).  We temporarily replace them with no-ops.
_INIT_FNS_TO_SKIP = (
    "uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_",
    "eye_", "dirac_", "xavier_uniform_", "xavier_normal_",
    "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_",
)


def _try_fast_init(config: dict, state: dict, **kwargs):
    """
    Construct the model on CPU but skip the expensive random weight
    initialisation (which the checkpoint overwrites anyway), then fill it from
    ``state`` via copy_.  Unlike a meta-device construction this keeps every
    computed constant (RoPE freqs, positional embeddings, etc.) as real tensors,
    so only the throwaway RNG is skipped.

    Returns the loaded model, or None if the fast path is not safe (the
    checkpoint is missing keys, which would leave uninitialised tensors behind)
    so the caller can fall back to the standard path.
    """
    import torch

    _init = torch.nn.init
    _saved = {name: getattr(_init, name) for name in _INIT_FNS_TO_SKIP
              if hasattr(_init, name)}

    def _noop(tensor, *args, **kwargs):
        return tensor

    try:
        for name in _saved:
            setattr(_init, name, _noop)
        model = __getattr__(config['name'])(**config['args'], **kwargs)
    except Exception:
        return None
    finally:
        for name, fn in _saved.items():
            setattr(_init, name, fn)

    # copy_ casts each checkpoint tensor to the destination parameter dtype,
    # exactly mirroring the standard load_state_dict path.
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        # A missing key means a parameter/buffer was never initialised by the
        # checkpoint and the RNG that would have set it was skipped — not safe.
        return None
    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
