import os
from typing import Union
import torch
from pixal3d.modules.sparse import SparseTensor, sparse_cat
from pixal3d.modules.sparse.basic import VarLenTensor, varlen_cat


_CFG_BATCH = os.environ.get("CFG_BATCH", "1") == "1"


def _cat_cond(cond_a, cond_b):
    """Concatenate two conditioning objects along the batch dimension.

    Handles nested dicts/tuples, SparseTensor, VarLenTensor, and plain
    torch.Tensor.  SparseTensor must be checked before VarLenTensor because it
    is a subclass.
    """
    if isinstance(cond_a, dict):
        return {k: _cat_cond(cond_a[k], cond_b[k]) for k in cond_a}
    elif isinstance(cond_a, tuple):
        return tuple(_cat_cond(a, b) for a, b in zip(cond_a, cond_b))
    elif isinstance(cond_a, SparseTensor):
        return sparse_cat([cond_a, cond_b], dim=0)
    elif isinstance(cond_a, VarLenTensor):
        return varlen_cat([cond_a, cond_b], dim=0)
    elif isinstance(cond_a, torch.Tensor):
        return torch.cat([cond_a, cond_b], dim=0)
    else:
        raise TypeError(f"_cat_cond: unsupported cond type {type(cond_a)!r}")


def _batch_x_t(x_t):
    """Duplicate x_t along batch dim to create a batch-2 input."""
    if isinstance(x_t, SparseTensor):
        return sparse_cat([x_t, x_t], dim=0)
    return torch.cat([x_t, x_t], dim=0)


def _split_pred(pred, batch_size: int):
    """Split a batched prediction back into (pred_pos, pred_neg)."""
    if isinstance(pred, SparseTensor):
        indices = list(range(2 * batch_size))
        return pred[indices[:batch_size]], pred[indices[batch_size:]]
    return pred[:batch_size], pred[batch_size:]


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        elif _CFG_BATCH:
            batch_size = x_t.shape[0]
            x_t_batched = _batch_x_t(x_t)
            cond_batched = _cat_cond(cond, neg_cond)
            pred_batched = super()._inference_model(model, x_t_batched, t, cond_batched, **kwargs)
            pred_pos, pred_neg = _split_pred(pred_batched, batch_size)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
        else:
            pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
