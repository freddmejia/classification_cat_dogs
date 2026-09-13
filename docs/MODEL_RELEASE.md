# Model release process

Follow this every time a changed notebook or architecture produces a model that apps should use.

- The contract lives in [MODEL_CONTRACT.md](MODEL_CONTRACT.md).
- Published releases live in `release/vX.Y.Z/`.
- Apps (such as the KMP app) read releases from that folder and never write to it.

## 1. Decide the release type

Answer in order; the first "yes" decides the type.

| Question | If yes |
|---|---|
| Did anything in the **input or output** change? Shape, color order, value range, normalization, output shape or meaning, classes, threshold | **Type B → MAJOR** (`2.0.0`) |
| Did the **runtime** change? Architecture, operators list, Flex ops, quantization type, model size (more than ±20%), GPU compatibility | **Type C → MINOR** (`1.1.0`) |
| Only the **weights** changed? Retraining, more epochs or data, fine-tuning settings | **Type A → PATCH** (`1.0.1`) |

A change that is both B and C is released as **B** (MAJOR); list the runtime changes in its notes too.

## 2. Checklist (in this project)

1. **Clean run:** Kernel → Restart Kernel and Run All Cells, then confirm execution counts increase from top to bottom.
2. **Quality gate:**
   - Record test precision, recall and accuracy, and the best val_loss.
   - Compare with the current release: accuracy 0.9435, precision 0.9503, recall 0.936.
   - Don't release a worse model unless there's a stated reason (for example, smaller or faster).
3. **Export,** with the version in the file names:
   - `cat_dog_vX.Y.Z.keras`
   - `cat_dog_vX.Y.Z.tflite`
   - `cat_dog_vX.Y.Z_optimized.tflite`
   - Core ML `.mlpackage`, only if an app uses it
4. **Verify the export** (section 3): the TFLite output matches Keras on `dog.png` and `cat.jpg`, the operators list, GPU compatibility, input/output details, and SHA-256 checksums.
5. **Create `release/vX.Y.Z/`** containing:
   - the `.tflite` files (and `.mlpackage` if any);
   - `dog.png`, `cat.jpg` (or new reference images);
   - `MODEL_CONTRACT.md`, copied after updating it;
   - `RELEASE_NOTES.md`, from the template in section 4.
6. **Update [MODEL_CONTRACT.md](MODEL_CONTRACT.md):** version, files, checksums, input/output, reference outputs, runtime requirements, changelog row.
7. **Update `CONTEXT.md`:** add a row to the results history.
8. **Never modify a published release folder.** Fixes go in a new version.
9. **Tell the app session:** "New model release vX.Y.Z (type A/B/C) in `release/vX.Y.Z/`. Follow its RELEASE_NOTES.md."

## 3. Export verification

Run in a throwaway container. The project is mounted read-only, and the running JupyterLab container isn't touched. Set `VERSION` and the file names first.

```bash
MSYS_NO_PATHCONV=1 docker run --rm -i -v "C:/Users/Usuario/Documents/ai/classification/cat_dogs:/workspace:ro" -w /workspace -e TF_CPP_MIN_LOG_LEVEL=3 -e CUDA_VISIBLE_DEVICES="" cat-dogs-tf:latest python - <<'PY'
import collections
import contextlib
import hashlib
import io
import re
import cv2
import numpy as np
import tensorflow as tf

KERAS = "cat_dog_mobilenetv3.keras"
TFLITES = ["cat_dog_mobilenetv3.tflite", "cat_dog_mobilenetv3_optimized.tflite"]
IMAGES = ["dog.png", "cat.jpg"]
SIZE = (128, 128)

keras_model = tf.keras.models.load_model(KERAS)
for path in TFLITES:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tf.lite.experimental.Analyzer.analyze(model_path=path, gpu_compatibility=True)
    ops = collections.Counter(re.findall(r"Op#\d+\s+(\w+)\(", buf.getvalue()))
    gpu = "compatible" if "looks compatible with GPU delegate" in buf.getvalue() else "check analyzer output"
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0], interp.get_output_details()[0]
    print(path, "| sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
    print("  ops", dict(sorted(ops.items())), "| flex", [k for k in ops if k.startswith("Flex")] or "none", "| gpu", gpu)
    print("  input", i["shape"].tolist(), i["dtype"].__name__, "| output", o["shape"].tolist(), o["dtype"].__name__)
    for name in IMAGES:
        rgb = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(tf.image.resize(rgb, SIZE).numpy().astype(np.float32), 0)
        interp.set_tensor(i["index"], x)
        interp.invoke()
        t = float(interp.get_tensor(o["index"])[0][0])
        k = float(keras_model.predict(x, verbose=0)[0][0])
        print(f"  {name}: keras {k:.5f} | tflite {t:.5f} | diff {abs(k - t):.5f}")
PY
```

**Pass criteria:**
- The default `.tflite` differs from Keras by < 0.001.
- The optimized one differs by < 0.02.
- No Flex ops, unless the release is type C and the notes say so.

## 4. RELEASE_NOTES.md template

```markdown
# Model release vX.Y.Z

| | |
|---|---|
| Date | YYYY-MM-DD |
| Release type | A (weights) / B (input/output) / C (runtime) |
| Contract version | X.Y.Z |
| Previous release | vA.B.C |
| Notebook commit | <git short hash> |

## What changed
- ...

## Metrics (test set, 2,000 images)
| Metric | Previous | This release |
|---|---|---|
| Accuracy | | |
| Precision | | |
| Recall | | |
| Best val_loss | | |

## Files
| File | Size | SHA-256 |
|---|---|---|

## Reference outputs
| Image | Keras | tflite | tflite optimized | Label |
|---|---|---|---|---|

## Input/output changes (type B only)
| Property | Before | After |
|---|---|---|

## Runtime changes (type C, or B with runtime changes)
- Operators added/removed:
- Flex ops:
- GPU compatibility:
- Size:

## What the app must do
- Type A: replace the model files, update checksums in the app's contract copy, run the reference test (±0.02).
- Type B: update preprocessing/postprocessing on Android and iOS, update tests, then do everything from type A.
- Type C: re-verify runtime support (LiteRT, TFLite/Core ML on iOS), GPU, and latency on real devices; update RESEARCH.md; then do everything from type A.
```

## 5. What moves between projects

| Item | Moves? | How |
|---|---|---|
| Model files, reference images | Yes | App copies from `release/vX.Y.Z/` |
| `MODEL_CONTRACT.md` | Yes (copy) | App replaces its copy; this project stays the source of truth |
| `RELEASE_NOTES.md` | Yes (read) | App session follows "What the app must do" |
| Library research (LiteRT, CameraX, iOS runtimes) | No | Lives in the app project (`docs/RESEARCH.md`); the app updates it for type C releases |
| Model-side research (conversion, quantization) | Summary only | Summarized in the release notes |
| Claude memories | **Never** | Machine- and project-specific. Lessons that matter to both sides go into the contract or `CLAUDE.md` |
