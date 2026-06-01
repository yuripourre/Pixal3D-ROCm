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
    """
    import struct
    import json
    import mmap
    import torch

    DTYPE_MAP = {
        "F64": (torch.float64, 8),
        "F32": (torch.float32, 4),
        "F16": (torch.float16, 2),
        "BF16": (torch.bfloat16, 2),
        "I64": (torch.int64, 8),
        "I32": (torch.int32, 4),
        "I16": (torch.int16, 2),
        "I8": (torch.int8, 1),
        "U8": (torch.uint8, 1),
        "BOOL": (torch.bool, 1),
    }
    COMPLEX_MAP = {
        "C64": (torch.float32, 4),
        "C128": (torch.float64, 8),
    }

    try:
        from safetensors.torch import load_file
        return load_file(filename, device=device)
    except Exception:
        pass

    tensors = {}
    with open(filename, "rb") as f:
        raw_header_len = f.read(8)
        header_len = struct.unpack("<Q", raw_header_len)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data_offset = 8 + header_len

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            abs_start = data_offset + start
            abs_end = data_offset + end
            raw = mm[abs_start:abs_end]

            if dtype_str in COMPLEX_MAP:
                float_dtype, itemsize = COMPLEX_MAP[dtype_str]
                buf = torch.frombuffer(bytearray(raw), dtype=float_dtype)
                t = torch.view_as_complex(buf.reshape(shape + [2]))
            elif dtype_str in DTYPE_MAP:
                torch_dtype, _ = DTYPE_MAP[dtype_str]
                t = torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape)
            else:
                raise ValueError(f"Unknown dtype {dtype_str!r} for tensor {key!r}")

            tensors[key] = t.to(device)

        mm.close()

    return tensors


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
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

    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(state, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
