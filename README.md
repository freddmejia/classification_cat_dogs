# Cat and Dog Classification with Transfer Learning

A notebook-based project for training, evaluating and exporting cat/dog classifiers with TensorFlow and Keras. It follows the process from an initial MobileNetV3Small experiment to animal-focused training images, a lightweight EfficientNetB0 classifier, and separate evaluation of transfer learning and fine-tuning.

## Motivation and starting point

The starting point was [Kavindutharaka's cat/dog classification notebook on Hugging Face](https://huggingface.co/Kavindutharaka/cat_dog_classifier/blob/main/cat_dog_classifier.ipynb), which is credited in the original training notebook. This project builds on that learning exercise to investigate practical questions:

- Can an ImageNet-pretrained backbone provide useful accuracy with a small model?
- Does fine-tuning improve both classes, rather than only the overall score?
- How do animal size, background and image framing affect predictions?
- Can the trained model be exported to TensorFlow Lite with consistent preprocessing and predictions?

Experiments with imported phone images and zoomed views motivated the work on framing, augmentation and YOLO-assisted cropping. Those observations are a reason to investigate, not proof that cropping or additional training always improves accuracy. The separate phase reports are intended to make those improvements, or regressions, measurable.

## The three training notebooks to watch

**Pay particular attention to these three files. They represent different stages of the project and do not have interchangeable settings or outputs.**

| Notebook | Role | Backbone and input | What to check |
|---|---|---|---|
| [CatDogClassifier TransferLearning.ipynb](CatDogClassifier%20TransferLearning.ipynb) | Original baseline and first mobile export | MobileNetV3Small, 128 x 128 | Original training data, initial evaluation and the published v1 model contract |
| [CatDogClassifier TransferLearningV2DataAugmentation.ipynb](CatDogClassifier%20TransferLearningV2DataAugmentation.ipynb) | Framing, zoom and cropped-data experiments | MobileNetV3Small, 224 x 224 | `train_crop075`, augmentation wiring, output paths and differences from the baseline |
| [CatDogClassifier Top5Ligero 01 EfficientNetB0.ipynb](CatDogClassifier%20Top5Ligero%2001%20EfficientNetB0.ipynb) | Current workflow with separate phase artifacts and extended metrics | EfficientNetB0, 224 x 224 | Validation curves, transfer/fine-tuning comparison, saved models and per-class metrics |

Start with the **EfficientNetB0 notebook** for the current end-to-end training workflow. Read the other two to understand the experiments that led to it. The EfficientNetB0 binary classifier has approximately **4.05 million total parameters**, including frozen parameters; its ImageNet classification head is replaced with a single sigmoid output.

Supporting files:

| File | Purpose |
|---|---|
| [CleanDataset.ipynb](CleanDataset.ipynb) | Detect animals with YOLO, create square crops and separate images with no detection |
| [ModelEvaluation.ipynb](ModelEvaluation.ipynb) | Evaluate a saved Keras model, inspect class imbalance and check release targets |
| [src/evaluation.py](src/evaluation.py) | Shared prediction collection, per-class reports and confusion-matrix plots |
| [Dockerfile](Dockerfile), [docker-compose.yaml](docker-compose.yaml), [env.yml](env.yml) | Reproducible notebook environment and dependency configuration |
| [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md) | Input/output and preprocessing contract for the original mobile model |
| [docs/MODEL_RELEASE.md](docs/MODEL_RELEASE.md) | Model release procedure and evaluation criteria |
| [release/v1.0.0/RELEASE_NOTES.md](release/v1.0.0/RELEASE_NOTES.md) | Recorded baseline results and published TFLite artifacts |

### Historical notebook details

The older notebooks retain experimental cells. Check the actual model graph before attributing a result to a feature:

- The original notebook instantiates Xception in an exploratory cell, but the classifier actually uses **MobileNetV3Small**. Its augmentation object is not connected to the classifier.
- The V2 notebook computes `x = data_augmentation(inputs)` but then calls `MobileNetV3Small(inputs, training=False)`. In that saved code, the backbone bypasses the augmented tensor. The EfficientNetB0 notebook correctly passes the augmented `x` into its backbone.
- In the original notebook, the image-inference cell loads `cat_dog_mobilenetv3.keras` before the following save cell. On a fresh run, save the newly trained model before executing that load/inference cell; otherwise it needs a pre-existing file and can load an older model.
- For V2, create the `models/` directory before saving if it does not exist. Its configuration and the original notebook use different paths and input resolutions.

## Dataset

The data comes from [Cats and Dogs for Classification by Dinesh Piyasamara on Kaggle](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification). The download URL is also recorded in the original training notebook.

Download and extract the dataset separately. Images are not distributed as part of the training code. Keep the following structure relative to the repository root:

```text
content/
  cats-and-dogs-for-classification/
    cats_dogs/
      train/
        cats/
        dogs/
      test/
        cats/
        dogs/
      train_crop075/        # Generated with CleanDataset.ipynb
        cats/
        dogs/
```

The recorded original dataset contains **8,000 training images** (4,000 per class) and **2,000 test images** (1,000 per class). The training notebooks reserve 10% of their selected training directory for validation with seed 42. The cropped directory is a derived dataset, so it has a different image count.

You can download the archive from Kaggle manually, or run this in a notebook after configuring Kaggle authentication for `opendatasets`:

```python
import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification",
    data_dir="content",
)
```

Check the extracted directory structure before running the loaders. A Kaggle login/API credential may be required; keep credentials in local configuration and out of notebooks and commits. Starting Docker does not download the dataset.

## How Docker is built

The [Dockerfile](Dockerfile) starts from `mambaorg/micromamba:2.9.0-debian13` and installs [env.yml](env.yml) into the container's **base** environment. Its configured stack includes:

| Component | Configuration in the repository |
|---|---|
| Python | 3.12 |
| TensorFlow | `tensorflow[and-cuda]==2.21.*` |
| PyTorch / torchvision | `2.11.0+cu128` / `0.26.0+cu128` |
| YOLO package | `ultralytics-opencv-headless==8.4.154` |
| Notebook tools | JupyterLab and ipywidgets |
| Data and images | NumPy, pandas, Matplotlib, opendatasets and headless OpenCV |

The build configures NVIDIA library paths and performs an import check. Its default command starts JupyterLab in `/workspace` on port 8888. The project is mounted at runtime rather than copied into the image.

Docker Compose defines:

- Service `notebook`, container `cat_dogs`, image `cat-dogs-tf:latest`.
- The repository mounted at `/workspace`, so notebook edits and generated artifacts persist on the host.
- Jupyter exposed at `127.0.0.1:8888`.
- Persistent `keras-cache` and `nv-cache` volumes for pretrained weights and CUDA compilation caches.
- NVIDIA GPU access, TensorFlow GPU memory growth and 2 GB of shared memory.
- Local environment variables loaded from `.env`.

The supplied Compose configuration expects an NVIDIA-capable Docker host. Follow [Docker's GPU access requirements](https://docs.docker.com/compose/how-tos/gpu-support/) for your platform. A CPU-only run requires removing the GPU device reservation in a local Compose configuration; the supplied configuration does not automatically fall back when Docker cannot provide a GPU.

## Run the project

### 1. Start JupyterLab

Install Docker with Compose and configure GPU access. Open a terminal at the repository root. In PowerShell, create `.env` only if it is missing:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Set your local credentials as needed. You can also add `JUPYTER_TOKEN=your-local-token` to `.env`; the Compose fallback token is `catdogs`.

```shell
docker compose up --build -d
docker compose logs -f notebook
```

Open [JupyterLab at localhost:8888](http://localhost:8888) and enter your configured token. Run notebooks from `/workspace`, the project root.

Check that the environment and GPU are visible:

```shell
docker compose exec notebook /opt/conda/bin/python -c "import tensorflow as tf; import torch; print('TensorFlow:', tf.__version__); print('TF GPUs:', tf.config.list_physical_devices('GPU')); print('PyTorch CUDA:', torch.cuda.is_available())"
```

Use `/opt/conda/bin/python` for direct container commands because a plain `docker exec ... python` may not activate the Micromamba environment. To stop the services after finishing:

```shell
docker compose down
```

The source mount and named cache volumes remain available for the next run.

### 2. Prepare animal-focused training images with YOLO

Open [CleanDataset.ipynb](CleanDataset.ipynb). It uses `yolov8n.pt` to detect COCO classes **15 (cat)** and **16 (dog)** with confidence threshold **0.25**. The detector weights are downloaded on first use if they are not available locally.

For each image, the notebook:

1. Keeps the highest-confidence cat/dog detection.
2. Calculates a square crop around its bounding box.
3. Writes the crop while preserving the original class subdirectory.
4. Copies images with no detection into `out/train_no_animal/` for inspection.
5. Records bounding boxes and detection metadata in `out/bboxes_train.csv`.

`COVERAGE = 0.75` targets an animal bounding-box longest side equal to 75% of the crop side. This is a **linear size target**, not 75% of the image area. Image boundaries can prevent the requested framing, so the notebook reports achieved coverage and capped crops.

**Align the output path before running.** The cleaning notebook defaults to `out/train_crop075`, while V2 and EfficientNetB0 expect `content/.../cats_dogs/train_crop075`. To create the data directly where those notebooks expect it, set these values in the cleaning configuration cell:

```python
SRC = Path("content/cats-and-dogs-for-classification/cats_dogs/train")
OUT_CROP = SRC.parent / "train_crop075"
COVERAGE = 0.75
LIMIT = 0
```

Run the model-loading, crop-helper, dataset-building and preview cells. For a small first check, use `LIMIT = 200`; set it back to `0` and rebuild before a complete training run. The optional coverage-sweep cell runs additional experiments at 0.50, 0.75 and 0.90 and writes its own directories under `out/`.

The recorded full cleaning run processed 8,000 images, created **7,050 crops** and separated **950 images with no detection**. These are historical run counts, not guaranteed results for another detector or dataset version.

YOLO is a preprocessing aid here, not the final binary classifier. It does not relabel the source folders or verify that a detection agrees with the folder label. A missing detection does not prove the image contains no animal. Review the crops, class balance and rejected images. Only training images are processed by this notebook; the test directory remains unchanged.

### 3. Train and fine-tune EfficientNetB0

Open [CatDogClassifier Top5Ligero 01 EfficientNetB0.ipynb](CatDogClassifier%20Top5Ligero%2001%20EfficientNetB0.ipynb) and review its configuration before running cells in order:

| Setting | Default |
|---|---|
| Training directory | `train_crop075` |
| Image size / batch size | 224 x 224 / 32 |
| Validation fraction / seed | 0.10 / 42 |
| Head-training epochs | Up to 100, Adam at 1e-3 |
| Fine-tuning epochs | Up to 50, Adam at 1e-5 |
| Fine-tuned backbone layers | Last 20, with BatchNormalization frozen |
| EarlyStopping patience | 15 for transfer learning, 5 for fine-tuning |
| Checkpoint criterion | Lowest `val_loss` within each phase |
| Decision threshold | 0.5 |
| TFLite export | Disabled unless `EXPORT_TFLITE=True` |

The model applies horizontal flips and zoom during training, followed by the ImageNet-pretrained backbone, global average pooling, dropout and a sigmoid classification head. The backbone includes preprocessing, so inputs are RGB pixel values in **[0, 255]**.

The notebook trains the head with the backbone frozen, then saves and evaluates that model **before** unfreezing layers for fine-tuning. It subsequently saves and evaluates the fine-tuned model under another name. Both phases use validation data and retain their own best weights through EarlyStopping. Neither phase automatically replaces the other as the chosen result.

For a short workflow check, set both epoch limits to 1. This exercises the pipeline but does not establish model quality. ImageNet weights need network access on first use. Reduce the batch size if memory is insufficient.

### 4. Inspect the two phase results

The EfficientNetB0 notebook records **loss, accuracy, precision, recall, ROC-AUC and F1** during training and validation, then evaluates each saved phase on the same test images. It also calculates per-class precision/recall/F1 and macro averages.

The single-output `f1_score`, precision and recall refer to the positive class, `class_names[1]` (normally `dogs`). `f1_score` uses an explicit 0.5 threshold. `macro_f1` averages the separate cat and dog F1 scores. ROC-AUC uses predicted probabilities across thresholds. See the [Keras metrics documentation](https://keras.io/api/metrics/classification_metrics/).

Generated paths use `cat_dog_EfficientNetB0_Top5Light` as the run name:

```text
models/top10/cat_dog_EfficientNetB0_Top5Light/
  cat_dog_EfficientNetB0_Top5Light_transfer_learning.keras
  cat_dog_EfficientNetB0_Top5Light_fine_tuning.keras
  cat_dog_EfficientNetB0_Top5Light_transfer_learning.weights.h5
  cat_dog_EfficientNetB0_Top5Light_fine_tuning.weights.h5
  labels.json

out/top10/cat_dog_EfficientNetB0_Top5Light/
  transfer_learning/
    metrics.json
    per_class.csv
    predictions.csv
    history.csv
    training_history.png
    confusion_matrix.png
  fine_tuning/              # Same report files for this phase
  stage_comparison.csv
  stage_comparison.png
```

Phase checkpoints are stored separately under `checkpoints/top10/<run-name>/`. The comparison table shows fine-tuning changes in percentage points, including negative changes. Keep either saved model based on the results you need. Re-running the same configuration overwrites that run's artifacts, so copy or rename a run before another experiment if you want to retain it.

### 5. Audit a saved model with ModelEvaluation

[ModelEvaluation.ipynb](ModelEvaluation.ipynb) uses [src/evaluation.py](src/evaluation.py) for a more detailed class-level audit. Its defaults target the original MobileNetV3Small model at 128 x 128. To inspect an EfficientNetB0 phase, change its configuration, for example:

```python
MODEL_PATH = "models/top10/cat_dog_EfficientNetB0_Top5Light/cat_dog_EfficientNetB0_Top5Light_fine_tuning.keras"
IMAGE_SIZE = (224, 224)
TEST_DIR = "content/cats-and-dogs-for-classification/cats_dogs/test"
BATCH_SIZE = 32
PREVIOUS = None
```

Use the corresponding `_transfer_learning.keras` path to audit the other phase. Match `IMAGE_SIZE` to the selected model. This evaluation notebook reports:

- Class support, correct predictions, precision, recall and F1.
- Overall accuracy, balanced accuracy and macro F1.
- Approximate 95% recall ranges, the recall gap between classes and a gap-versus-noise check.
- A confusion matrix with counts and row percentages.
- A release-gate table with default targets: recall of at least 0.93 per class, recall gap at most 0.03, and test-set class-count ratio at least 0.90.

Set `PREVIOUS` to a dictionary of previous **per-class recall values**, such as `{"cats": 0.951, "dogs": 0.936}`, to also check for drops greater than 0.01. The uncertainty calculation is a simple normal approximation, not a definitive significance test. A release-gate pass is a project criterion, not a guarantee on unseen phone photos.

The standalone evaluation notebook currently does **not** calculate ROC-AUC; AUC and the training/validation F1 histories are part of the EfficientNetB0 training notebook.

## Recorded baseline results

The outputs in [ModelEvaluation.ipynb](ModelEvaluation.ipynb) and the [v1.0.0 release notes](release/v1.0.0/RELEASE_NOTES.md) record the following results for the original **128 x 128 MobileNetV3Small** model on 2,000 test images:

| Class | Support | Correct | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Cats | 1,000 | 951 | 95.10% | 0.9369 | 0.9439 |
| Dogs | 1,000 | 936 | 93.60% | 0.9503 | 0.9431 |

Overall accuracy was **94.35%**, balanced accuracy **94.35%**, and macro F1 approximately **0.9435**. The recorded class recall gap was 1.5 percentage points, with 49 cats predicted as dogs and 64 dogs predicted as cats. The release gate passed for that run.

These are historical baseline results, **not EfficientNetB0 results**. Run the current notebook to obtain its metrics. Comparisons should account for changes in training data, resolution and preprocessing. If the test results are used to choose a model repeatedly, use fresh held-out data for a final unbiased assessment.

## Common checks before a run

| Symptom | Check |
|---|---|
| Dataset directory not found | Extraction layout, project working directory and the `OUT_CROP`/training-path alignment |
| Missing `.keras` model | Train and save first, or use the included TFLite release for the no-training example |
| Wrong input shape | 128 x 128 for the original release; 224 x 224 for V2 and EfficientNetB0 |
| Mobile prediction disagrees with the notebook | Model version, RGB order, pixel range, resize/crop policy and output-to-label mapping |
| Out of memory | Lower `BATCH_SIZE` and restart the notebook kernel before another model run |
| GPU not visible | Host driver, Docker GPU access and the runtime check above |
| Slow first GPU operation | Initial CUDA kernel compilation can be lengthy on newer GPU architectures; inspect logs before restarting |

Training outputs, datasets, checkpoints and downloaded detector weights are generated locally. The versioned release directory is a separate snapshot of the baseline model; it is not automatically updated by a notebook run.
