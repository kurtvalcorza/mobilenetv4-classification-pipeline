# Weight provenance and DIMER hosting

- Upstream: `timm/mobilenetv4_conv_small.e2400_r224_in1k`
- Immutable revision: `331fb803779522b685cf942e15f914fb6741c1eb`
- Weight format: SafeTensors (`model.safetensors`, 15223016 bytes)
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/mobilenetv4-conv-small/` with `dimer-base-manifest.json` (per-file bytes + SHA-256, `totalBytes` 15236763); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/mobilenetv4_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license. Upstream notes these `timm`-trained weights are the only known MobileNetV4 weights (the official TensorFlow weights are unreleased).
- Loader trust boundary: `timm==1.0.29` built-in `mobilenetv4_conv_small` architecture; weights loaded from a file path via `pretrained_cfg_overlay`; no remote code is executed. Hub download is opt-in and pinned to the revision above.
