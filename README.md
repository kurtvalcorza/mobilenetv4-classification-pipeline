# MobileNetV4 Classification Pipeline

DIMER inference wrapper for **`timm/mobilenetv4_conv_small.e2400_r224_in1k`** — ImageNet-1k image classification (1000 classes), edge/CPU profile — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `timm/mobilenetv4_conv_small.e2400_r224_in1k` (MobileNetV4-Conv-Small, 3.8 M parameters)
- Revision: `331fb803779522b685cf942e15f914fb6741c1eb`
- Upstream weight license: Apache-2.0
- Upstream task: image classification, 1000 ImageNet-1k classes, 224×224 eval input
- Repository adaptation: **none**; inference only

## Quick start

```python
from PIL import Image
from mobilenetv4_classification_pipeline import MobileNetV4ClassificationPipeline, top_k_accuracy

pipe = MobileNetV4ClassificationPipeline.from_pretrained(device="cpu")   # omit device= to use cuda:0 when present
result = pipe.predict(Image.open("photo.jpg"), top_k=5)
print(result["predictions"][0]["predicted_label"], result["predictions"][0]["top_k"][0]["score"])
print(top_k_accuracy(result["predictions"], [target_index], k=1))
```

`score` is a softmax score over 1000 classes, not a calibrated probability; the reported label is the argmax. Measured on CPU (Intel Core Ultra 9 275HX): load 1.90 s, one prediction 0.19 s — see `MODEL_CARD.md` → Runtime.

## Weights layout

```
weights/mobilenetv4-conv-small/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # timm pretrained_cfg: input size, mean/std, crop
  model.safetensors          # 15223016 bytes, git-ignored
```

`from_pretrained()` calls `verify_snapshot()` first and refuses to load if any file is missing or its SHA-256 differs from the manifest. Without a snapshot, `allow_download=True` loads from the Hub through timm's `hf-hub:timm/mobilenetv4_conv_small.e2400_r224_in1k@331fb803779522b685cf942e15f914fb6741c1eb` form; the default is to refuse. To stage the snapshot: `hf download timm/mobilenetv4_conv_small.e2400_r224_in1k --revision 331fb803779522b685cf942e15f914fb6741c1eb --local-dir weights/mobilenetv4-conv-small`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed
```

Smoke (loads the verified snapshot on CPU and classifies one synthetic image):

```python
from PIL import Image
from mobilenetv4_classification_pipeline import MobileNetV4ClassificationPipeline

pipe = MobileNetV4ClassificationPipeline.from_pretrained(device="cpu")
print(pipe.predict(Image.new("RGB", (256, 256), (90, 140, 200)))["predictions"][0]["predicted_label"])
```

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.0 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md`.
