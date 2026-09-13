# CLAUDE.md

This file gives Claude Code guidance for working in this repository.

## Project status

This project is a deep learning image classifier that labels a photo as **cat** or **dog**, built with TensorFlow/Keras and transfer learning.

As of 2026-09-13 the repo holds only the environment setup: `Dockerfile`, `docker-compose.yaml`, `env.yml`, `.dockerignore`, and `.gitignore`. There is no training code or data yet, and the folder is not a git repository.

The reference implementation is the Colab notebook https://huggingface.co/Kavindutharaka/cat_dog_classifier/blob/main/cat_dog_classifier.ipynb. The first milestone is a local version of it.

Scope rule: work only inside this folder (`C:\Users\Usuario\Documents\ai\classification\cat_dogs`). Do not read or modify sibling or parent directories.

## Host environment (verified)

- OS: Windows 11 with WSL2; shell PowerShell 5.1 (Git Bash also available)
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, driver 595.97 (Blackwell, compute capability 12.0)
- Docker Desktop 28.3.3, Compose v2.39.2, with the `nvidia` container runtime registered
- Host Python 3.12.0 has no ML packages. **Run everything inside the container.** TensorFlow GPU does not run on native Windows.

## Environment files

| File | Purpose |
|---|---|
| `env.yml` | Conda/micromamba env: python 3.12, jupyterlab, and ipywidgets from conda-forge; `tensorflow[and-cuda]==2.21.*`, numpy, pandas, matplotlib, opendatasets, and opencv-python-headless from pip |
| `Dockerfile` | `mambaorg/micromamba:2.9.0-debian13`; installs `env.yml` into the base env; sets `LD_LIBRARY_PATH` to the pip `nvidia/*/lib` folders (required for the GPU, see GPU notes); fails the build if any notebook import breaks |
| `docker-compose.yaml` | Service `notebook`: GPU reservation, JupyterLab on `127.0.0.1:8888`, `env_file: .env`, project bind-mounted at `/workspace`, named volumes `keras-cache` (ImageNet weights) and `nv-cache` (JIT-compiled CUDA kernels) |
| `.env` / `.env.example` | Runtime secrets (`KAGGLE_API_TOKEN`). `.env` is git-ignored and docker-ignored, and Compose requires it to exist; copy `.env.example` to create it |

Add new dependencies to `env.yml`, then rebuild. Put anything that has to stay version-compatible with TensorFlow (such as numpy) in the `pip:` section, so one resolver handles it.

## Commands

```powershell
docker compose build                     # rebuild after changing env.yml
docker compose up -d                     # JupyterLab at http://127.0.0.1:8888 (token: $JUPYTER_TOKEN, default "catdogs")
docker compose logs -f notebook
docker compose exec notebook python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
docker compose exec notebook python path/to/script.py
docker compose down
```

## GPU notes (RTX 5060 Ti, Blackwell)

Verified on 2026-09-13 with TF 2.21.0, cuDNN 9.26, and driver 595.97:
- The TF 2.21 wheels only include CUDA kernels up to compute capability 9.0. On this 12.0 GPU, TF logs `not built with CUDA kernel binaries compatible with compute capability 12.0a ... jit-compiled from PTX, which could take 30 minutes or longer`. **This is expected, and it works.** Measured: first GPU ops about 7 s; Xception training epoch 1 took 9.3 s (JIT and autotuning), epoch 2 took 0.3 s for 640 images.
- Without the Dockerfile's `LD_LIBRARY_PATH`, TF can't load `libcusolver.so.11`. It then prints `Cannot dlopen some GPU libraries` and **silently runs on the CPU** (`list_physical_devices('GPU')` returns `[]`). If the GPU list is ever empty, check this first.
- Compiled kernels are cached in `~/.nv/ComputeCache`, which is persisted in the `nv-cache` volume; `CUDA_CACHE_MAXSIZE` is 4 GB. Only the first run after clearing that cache is slow.
- `cuda_timer.cc ... Delay kernel timed out` errors during the first epoch are harmless autotuning noise.
- Don't change the TensorFlow version, or the `nvidia-*` package versions it pulls in, without re-running the GPU check above. The `LD_LIBRARY_PATH` folder list must match the installed packages.

## Data

- Dataset: Kaggle `dineshpiyasamara/cats-and-dogs-for-classification` (217 MB), downloaded with `opendatasets`.
- Layout after download: `cats-and-dogs-for-classification/cats_dogs/{train,test}/{cats,dogs}`. Train has 8,000 images (split 90/10 into train/val with `seed=42`); test has 2,000.
- **Kaggle credentials:** `KAGGLE_API_TOKEN` (`KGAT_...` format) lives in `.env` and reaches the container at runtime through `env_file`. **Never put it in the Dockerfile** (`ENV`/`ARG`), because it would stay in the image layers and in `docker history`.
  - The `kaggle` package (1.8+) logs in from `KAGGLE_API_TOKEN` on `import kaggle`, then removes the variable from `os.environ`. Reuse `kaggle.api`; a new `KaggleApi()` will not find the token.
  - `opendatasets` ignores `KAGGLE_API_TOKEN`. It only reads `./kaggle.json` from the current working directory, and otherwise prompts for a username and key before it imports `kaggle`. `kaggle.json` is git-ignored and docker-ignored; never commit it.
- Class indices come from `image_dataset_from_directory`, which sorts class folders alphabetically: `cats = 0`, `dogs = 1`. So sigmoid output > 0.5 means dog.

## Reference model (from the notebook)

- Input 128×128 RGB, batch size 32, pixels scaled with `x / 255`.
- Backbone: `tf.keras.applications.xception.Xception(include_top=False, weights="imagenet", pooling="max")`, frozen.
- Head: Flatten → Dense 128 relu → Dense 128 relu → Dense 32 relu → Dense 1 sigmoid.
- Training: Adam, BinaryCrossentropy, accuracy metric, 5 epochs. Evaluated on the test set with Precision, Recall, and BinaryAccuracy.

Known issues to fix when porting the notebook:
- `data_augmentation` is defined but never added to the model.
- Xception expects inputs in [-1, 1] (`tf.keras.applications.xception.preprocess_input`), not [0, 1].
- The hardcoded Colab paths (`/content/...`) must become paths relative to `/workspace`.
- `google.colab` can't be installed outside Colab: PyPI has no `google-colab` distribution, and adding it to `env.yml` breaks the build. Replace `files.upload()` with `ipywidgets.FileUpload` (installed; verified), and decode `upload.value[0]["content"]` with `cv2.imdecode`. An image path also works.
- The notebook overwrites the `time` module with a float (`time = end_time - start_time`).
- OpenCV loads images as BGR. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling predict.
- The model is never saved. Save it with `model.save("checkpoints/<name>.keras")`.

## Conventions

- Write all code in English, including identifiers and user-facing strings.
- Do not add comments in code.
- Reusable logic belongs in Python modules; notebooks are for exploration.
- Preprocessing at inference time must match training: same resize and same scaling.
- Keep data (`cats-and-dogs-for-classification/`, `data/`), `checkpoints/`, and `outputs/` out of git. `.gitignore` already covers them.
- The environment sets `TF_FORCE_GPU_ALLOW_GROWTH=true`. Don't hardcode GPU memory limits in code.
