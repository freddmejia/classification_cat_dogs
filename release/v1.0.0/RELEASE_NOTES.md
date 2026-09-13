# Model release v1.0.0

| | |
|---|---|
| Date | 2026-09-13 |
| Release type | Initial release (first published model) |
| Contract version | 1.0.0 |
| Previous release | none |
| Notebook commit | `4cf2fab` ("load weights"), `CatDogClassifier TransferLearning.ipynb` |

## What changed

The first model published for apps:
- MobileNetV3Small (ImageNet) transfer learning, 128×128 RGB input with raw 0–255 pixels, sigmoid output (dog).
- **Stage 1:** head trained on a frozen backbone. Best val_loss 0.1260 at epoch 36.
- **Stage 2:** last 20 backbone layers fine-tuned with BatchNormalization frozen, Adam 1e-5, EarlyStopping patience 5. Best val_loss 0.1185 at epoch 16.

## Metrics (test set, 2,000 images)

| Metric | Previous | This release |
|---|---|---|
| Accuracy | – | 0.9435 |
| Precision | – | 0.9503 |
| Recall | – | 0.9360 |
| Best val_loss | – | 0.1185 |

About 113 errors: roughly 64 dogs predicted as cats and 49 cats predicted as dogs.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `cat_dog_mobilenetv3.tflite` | 3,754,192 bytes | `cc4ae83dd5c2466a9143e012f4d5b94f8502064ab3b31a0da21b2eb8c94f0c18` |
| `cat_dog_mobilenetv3_optimized.tflite` | 1,131,360 bytes | `d11d65ea068c1b537ac936cc6ef67e0be9122dd8e266284dbd8f704c11550536` |
| `dog.png` | 84,520 bytes | `a364a3e2ff3bddadde85897aeccb8b3051d9cd6822b79db96fb6d0a4f69b59a1` |
| `cat.jpg` | 331,846 bytes | `9b46595fdf92b4a91d380e6b94721da5a68ec18ace40f4829a29db357d79bda5` |
| `MODEL_CONTRACT.md` | – | Snapshot of contract 1.0.0 |

## Reference outputs

| Image | Keras | tflite | tflite optimized | Label |
|---|---|---|---|---|
| `dog.png` | 0.99254 | 0.99254 | 0.99545 | dog |
| `cat.jpg` | 0.00225 | 0.00225 | 0.00266 | cat |

## Input/output

See `MODEL_CONTRACT.md`: input `[1,128,128,3]` float32 RGB, raw 0–255 (never divide by 255); output `[1,1]` float32 sigmoid, `p > 0.5` → dog.

## Runtime

- **Operators:** `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `HARD_SWISH`, `LOGISTIC`, `MEAN`, `MUL`. All built-in.
- **Flex ops:** none.
- **GPU compatibility:** compatible per the TFLite analyzer.
- **Size:** 3.58 MB default, 1.08 MB optimized (dynamic-range int8 weights, float input/output).

## What the app must do

- **Initial integration:**
  - Add `cat_dog_mobilenetv3.tflite` to the app resources.
  - Copy `MODEL_CONTRACT.md` to the app's `docs/`.
  - Implement preprocessing exactly as the contract says.
  - Pass the reference test: label matches, value within ±0.02 on `dog.png` and `cat.jpg`.
- **Not yet verified on devices:**
  - LiteRT 2.2.0 on Android (CPU and GPU).
  - The iOS runtime (Core ML conversion or the TensorFlowLiteObjC 2.17.0 pod).
  - Latency.
