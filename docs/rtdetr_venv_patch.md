# RT-DETR venv patch: `valid_mask` inference tensor crash

## Bug

RT-DETR training crashes on the very first batch with:

```
RuntimeError: Inference tensors cannot be saved for backward.
To work around you can make a clone to get a normal tensor and use it in autograd.
```

Traceback ends at:
```
ultralytics/nn/modules/head.py, line ~1672, in _get_decoder_input
    features = self.enc_output(self.valid_mask * feats)
```

## Root cause

`RTDETRDecoder._generate_anchors` is decorated with `@smart_inference_mode()` (effectively `@torch.inference_mode()`). During the AMP sanity check that runs before training, this method is called and stores `valid_mask` as an inference-mode tensor in `self.valid_mask`. When the training forward pass later tries to multiply `valid_mask` by `feats` and save the result for autograd, PyTorch 2.1 raises the error.

Upstream issue: https://github.com/ultralytics/ultralytics/issues/23359

## Affected versions

Confirmed broken in Ultralytics **8.4.35** and **8.4.40** with PyTorch **2.1.2+cu121**.

## Fix

In `.venv/lib/python3.11/site-packages/ultralytics/nn/modules/head.py`, find `_get_decoder_input` (around line 1672) and add `.clone()`:

```python
# Before (broken):
features = self.enc_output(self.valid_mask * feats)

# After (fixed):
features = self.enc_output(self.valid_mask.clone() * feats)
```

This clone is applied outside inference mode, producing a regular tensor that autograd can track.

## Re-applying after venv recreation

If `uv sync` or `uv venv` recreates the venv, reapply with:

```bash
sed -i 's/self\.enc_output(self\.valid_mask \* feats)/self.enc_output(self.valid_mask.clone() * feats)/' \
    .venv/lib/python3.11/site-packages/ultralytics/nn/modules/head.py
```
