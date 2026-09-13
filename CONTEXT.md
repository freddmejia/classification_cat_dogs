# Project context: cats vs dogs classifier

Snapshot of where the project stands, so work can continue in a new session. Last updated **2026-09-13, 13:40**.
For environment setup and conventions, see [CLAUDE.md](CLAUDE.md). This file covers the state of the work and why things are the way they are.

## 1. Goal

A binary image classifier (**cat** vs **dog**) built with **TensorFlow/Keras transfer learning** on MobileNetV3Small.

- **Starting point:** the Colab notebook https://huggingface.co/Kavindutharaka/cat_dog_classifier/blob/main/cat_dog_classifier.ipynb.
- **Porting:** it's now a local notebook that runs on the GPU in Docker.
- **Outputs:** the trained model is exported as Keras and TFLite files.

## 2. Environment (verified)

| Item | Value |
|---|---|
| Host | Windows 11 with WSL2, Docker Desktop 28.3.3, Compose v2.39.2 |
| GPU | RTX 5060 Ti, 16 GB, Blackwell (compute capability 12.0), driver 595.97 |
| Image | `cat-dogs-tf:latest` (18.8 GB) from `mambaorg/micromamba:2.9.0-debian13` + `env.yml` |
| Libraries | Python 3.12, TensorFlow 2.21.0, Keras 3.15.1, numpy 2.5.3, pandas 3.0.5, OpenCV headless, opendatasets, kaggle 2.2.4, JupyterLab 4.6.3, ipywidgets 8.1.9 |
| Service | `docker compose up -d` → JupyterLab at http://127.0.0.1:8888 (token `catdogs`) with the project mounted at `/workspace` |

**Environment fixes that must stay:**
- **`LD_LIBRARY_PATH` in the Dockerfile** points to the pip `nvidia/*/lib` folders. Without it, TF can't load `libcusolver.so.11` and silently runs on the **CPU**.
- **Slow first GPU run is expected.** TF 2.21 has no Blackwell (compute capability 12.0) kernels, so it compiles them at runtime: about 7–10 s on first use, then fast. The `nv-cache` volume is meant to keep the compiled kernels between restarts, but `~/.nv` is owned by root, so it isn't being used yet (see §9).
- **Kaggle credentials:** `KAGGLE_API_TOKEN` is in `.env` and loaded at runtime via `env_file`. It is never baked into the image.
  - `opendatasets` **ignores** the token and asks for a username and key.
  - Use `kaggle.api.dataset_download_files("dineshpiyasamara/cats-and-dogs-for-classification", path="content", unzip=True)` instead.
- **`google.colab` can't be installed locally.** Use `ipywidgets.FileUpload` or an image path.

## 3. Project files

| Path | What it is | In git? |
|---|---|---|
| `CatDogClassifier TransferLearning.ipynb` | Main notebook: data, 2-stage training, evaluation, export | yes |
| `Dockerfile`, `docker-compose.yaml`, `env.yml` | Environment | yes |
| `.env` / `.env.example` | Kaggle token (real) / placeholder | ignored / yes |
| `CLAUDE.md`, `CONTEXT.md` | Guidance / this context file | yes / new, not committed |
| `content/cats-and-dogs-for-classification/cats_dogs/{train,test}/{cats,dogs}` | Dataset | ignored |
| `checkpoints/stage1.keras` | Stage 1 model, saved before fine-tuning | ignored |
| `cat_dog_mobilenetv3.keras`, `.weights.h5` | Final model (after fine-tuning) | ignored |
| `cat_dog_mobilenetv3.tflite` (3.58 MB), `_optimized.tflite` (1.08 MB, `Optimize.DEFAULT`) | TFLite exports; input `(None, 128, 128, 3)` float32 raw 0–255 pixels, output `(None, 1)` sigmoid | **yes, committed** |
| `dog.png`, `cat.jpg` | Test images | yes |
| `docs/MODEL_CONTRACT.md` | **Source of truth** for the model interface (contract version 1.0.0) | new |
| `docs/MODEL_RELEASE.md` | Release process: A/B/C types, checklist, verification script, release note template | new |
| `release/v1.0.0/` | First published model release, read by apps (never modified after publishing) | new |
| `handoff/kmp-app/` | Starter files for the KMP app project; the user moves them out | new, not committed |
| `CLASSIFIER CAT DOG USING TRANSFER L.txt` | Personal notes and links. **Contains the real Kaggle token** (see §8) | **yes, committed** |

- **Git:** branch `main` tracks `origin/main` (github.com/freddmejia/classification_cat_dogs, private). Commits: `9a55770 upadte code` → `2ce3510 add transferlearning and fine tunning` → `c2d743c better results` → `4cf2fab load weights`.

## 4. Data

- **Kaggle dataset:** `dineshpiyasamara/cats-and-dogs-for-classification`.
- **Split:**
  - Train folder: 8,000 images, split 90/10 with `seed=42` → **7,200 train / 800 validation**.
  - Test folder: **2,000** images (1,000 per class).
- **Loading:** `image_dataset_from_directory`, 128×128, batch 32.
- **Classes:** `['cats', 'dogs']`, so a sigmoid output > 0.5 means **dog**.

## 5. Current pipeline (notebook as of 13:34, all cells run in order)

1. **Pixels:** raw 0–255, **no `/255`**. MobileNetV3 rescales internally (`include_preprocessing=True`).
2. **Model (functional API):** `MobileNetV3Small(include_top=False, weights="imagenet")` called with `training=False`, then GAP, Dropout(0.2), Dense(1, sigmoid).
3. **Stage 1, head only:** backbone frozen, Adam 1e-3, BCE, EarlyStopping(val_loss, patience 15, restore best), up to 100 epochs. Then `model.save("checkpoints/stage1.keras")`.
4. **Stage 2, fine-tune:**
   - Unfreeze the backbone, then re-freeze everything except the **last 20 layers**.
   - **Keep every BatchNormalization layer `trainable=False`.**
   - `fine_tune_callbacks`: EarlyStopping patience 5. Adam 1e-5, up to 50 epochs.
5. **Evaluation:** Precision, Recall and BinaryAccuracy over the test set; accuracy/loss curves side by side in one compact figure.
6. **Export:** `.keras`, `.weights.h5`, `.tflite`, optimized `.tflite`.

## 6. Results history (same validation split and test set)

| # | Setup | Best val_loss | Test precision | Recall | Accuracy | Errors / 2,000 |
|---|---|---|---|---|---|---|
| 0 | Original Colab-style: Xception, `/255` (stale numbers) | – | 0.9645 | 0.952 | 0.9585* | – |
| 1 | MobileNetV3 frozen, **`/255` double-scaled**, Flatten head | ~0.516 | – | – | val ~0.73–0.77 | – |
| 2 | Frozen, fixed scaling, augmentation (rotation ±72°), GAP head | 0.1469 | 0.9327 | 0.9150 | 0.9245 | 151 |
| 3 | Sequential, last 20 unfrozen, no augmentation, 40 epochs | 0.1366 | 0.9345 | 0.9270 | 0.9310 | 138 |
| 4 | Same as 3 with 100 epochs, stopped at 71, restored epoch 56 | 0.1462 | 0.9433 | 0.9310 | 0.9375 | 125 |
| 5a | Functional, frozen (stage 1) | 0.1273 | 0.9458 | 0.9250 | 0.9360 | 128 |
| 5b | + full backbone fine-tune, **BN not frozen** | 0.1342 | 0.9232 | 0.9380 | 0.9300 | 140 |
| 6a | **Current, stage 1** (best epoch 36, stopped at 51) | 0.1260 | 0.9456 | 0.9210 | 0.9340 | 132 |
| **6b** | **Current, stage 2** (last 20 layers, **BN frozen**, best epoch 16, stopped at 21) | **0.1185** | **0.9503** | **0.9360** | **0.9435** | **113** |

\* Row 0's test metrics were run in an earlier kernel state, before the MobileNet model shown with them was built, so they don't belong to any model above.

- **Best model:** **6b**, test accuracy 0.9435, about 64 dogs predicted as cats and 49 cats predicted as dogs.
- **`dog.png`:** 0.9923, predicted dog.

## 7. Key findings (don't repeat these mistakes)

1. **Double scaling.** Keras MobileNetV3 has `Rescaling(1/127.5, offset=-1)` built in and expects 0–255 input. Dividing by 255 first squeezes every image to about [-1.0, -0.99], and accuracy falls to ~0.75.
2. **Keras 3 BatchNorm gotcha (verified in this image).**
   - After `base.trainable = True`, **`base(inputs, training=False)` does NOT keep BatchNorm frozen during `fit`.** All 34 BN layers updated their statistics; training-mode accuracy was 0.827 vs 0.925 in inference mode.
   - **Fix:** set `layer.trainable = False` on every `BatchNormalization` layer, then recompile.
   - **Result:** 0 BN changes, and fine-tuning helps instead of hurting (row 5b → 6b).
   - Reference: keras-team/keras issue #20285.
3. **Validation above training** during stage 1 is dropout/augmentation, not a bug. Overfitting only starts when training loss drops below validation loss and validation loss rises (run 4, after epoch ~34).
4. **Strong augmentation hurts the frozen linear head.** `RandomRotation(0.2)` means ±72°. Rows 2 → 5a: val_loss 0.147 → 0.127 without it.
5. **Unfreezing the whole backbone on 7,200 images** made things worse even with BN handling aside; the last 20 layers is the better choice.
6. **Stale kernel state:** check that execution counts increase top to bottom before trusting metrics. Use Restart & Run All.
7. **128×128 input** loads ImageNet weights meant for 224×224 (Keras warns). It works, but 224 might transfer better (not tried yet).

## 8. Security: action needed

- **`CLASSIFIER CAT DOG USING TRANSFER L.txt` contains the real Kaggle API token.**
  - It was committed in `9a55770` and pushed to `origin` (a private GitHub repository).
  - Anyone with access to the repo can read it, and it stays in git history.
- **To do:**
  1. **Regenerate the token** at kaggle.com/settings/api, and put the new one only in `.env`.
  2. Remove the token line from the notes file (or git-ignore the file), and commit.
  3. Optionally purge it from history (`git filter-repo`) and force-push. This rewrites history, so decide deliberately.
- **Also:** `*.tflite` isn't in `.gitignore`, so the TFLite models are committed (about 4.7 MB). That's fine if intended.

## 9. Open next steps

- [ ] Rotate the Kaggle token and clean the notes file (§8).
- [ ] Compare the TFLite models against Keras on the test set (the optimized version may lose accuracy).
- [ ] Try `IMAGE_SIZE = (224, 224)` with the same two-stage recipe.
- [ ] Try light augmentation during fine-tuning only (`RandomFlip("horizontal")`, `RandomRotation(0.05)`).
- [ ] Fix `~/.nv` ownership in the image (`mkdir` + `chown` in the Dockerfile), so compiled GPU kernels persist. This needs a rebuild and a container restart; ask first.
- [ ] Remove the unused Xception model cell from the notebook.
- [ ] Ideas from the notes file: a PyTorch CNN (huggingface.co/ljt019/cat_dog_classifier_cnn), ViT fine-tuning (google/vit-base-patch16-224), other references (kaggle adinishad PyTorch notebook, huggingface 24f2004275/cat-dog-classifier).

## 10. Working rules agreed in this project

- Write code in English and **don't add comments in code** (CLAUDE.md).
- **Don't restart, recreate or stop the running `cat_dogs` container** without asking. Run checks in throwaway containers (`docker run --rm` or `docker compose run --rm`).
- **Before editing the notebook:**
  - back it up;
  - abort if the file changed since it was read;
  - edit cells by their ID;
  - clear the edited cells' outputs;
  - then the user reloads it from disk (File → Reload Notebook from Disk) **before saving**.
- When the user asks for **analysis only**, don't edit any file.
