# Codex handoff: inspect incorrect mobile cat/dog predictions

## Task and constraints

Inspect the mobile app to explain why an imported dog image is displayed as a cat when the user reports using the same image as in the training notebook.

This is an investigation only. Do not write or modify application code, notebooks, models, or configuration. Do not retrain or re-export the model. Report the evidence and the smallest proposed fix before implementation. Any future code must be in English and contain no comments.

Work inside the mobile project opened by the user and follow its repository instructions. If this file is opened in the ML project, inspect the available handoff material and request the mobile project location rather than searching unrelated directories. Do not restart or stop the user's running Jupyter container. Published model releases are read-only.

## User report and unknowns

- The user imported a dog image on mobile and the app displayed cat.
- The user says it is the same image used in the notebook.
- The user wants the mobile discrepancy investigated; do not assume poor training or recommend more epochs without evidence.
- The exact failing image, notebook score for that image, mobile raw score, device, OS, runtime, delegate, and bundled model version have not been established in this conversation.
- Project documentation describes a Kotlin Multiplatform app in a separate repository. Its implementation has not yet been inspected.

## ML project and evidence

ML project: `C:\Users\Usuario\Documents\ai\classification\cat_dogs`

Relevant files in that project:

- `CatDogClassifier TransferLearning.ipynb`: training, evaluation, prediction, and export.
- `docs/MODEL_CONTRACT.md`: source of truth for the model interface and recorded reference checks.
- `release/v1.0.0/`: published models, reference images, contract, and release notes.
- `CONTEXT.md`: training history and previous findings.
- `handoff/kmp-app/`: starter handoff material, not proof of the actual app implementation.

Recorded final Keras evaluation on 2,000 balanced test images:

| Metric | Value |
|---|---:|
| Accuracy | 0.9435 |
| Precision, dog positive | 0.9503 |
| Recall, dog positive | 0.9360 |

These are good aggregate results, not a guarantee for every image. They do not establish the cause of an app/notebook disagreement. The optimized TFLite model's full test-set accuracy has not been measured in the recorded contract.

The active classifier uses MobileNetV3Small with internal preprocessing, GlobalAveragePooling2D, Dropout(0.2), and Dense(1, sigmoid). The notebook also contains an unused Xception cell; do not infer the deployed architecture from that cell.

## Exact inference contract

| Property | Expected value |
|---|---|
| Input shape | `[1, 128, 128, 3]`, NHWC |
| Input type | `float32`, including the optimized model |
| Channels | RGB, interleaved row by row |
| Pixel range | Raw `0.0` to `255.0` |
| External normalization | None |
| Internal normalization | Rescaling by `1/127.5` with offset `-1` |
| Output shape and type | `[1, 1]`, `float32` |
| Output meaning | Single sigmoid dog score `p` |
| Label rule | `p > 0.5`: dog; otherwise cat |
| Displayed class score | Dog: `p`; cat: `1 - p` |

Do not divide inputs by 255 or normalize them to [-1, 1] in the app. The model already performs normalization. The contract records that feeding 0-1 pixels produced about 0.55 for both reference images. This is a diagnostic clue, not proof that the app currently makes this mistake.

Do not apply softmax or argmax to the single output. There are not two output entries for cat and dog. Do not apply sigmoid a second time.

Reference preprocessing: OpenCV decode, BGR-to-RGB conversion, resize the full image to 128 by 128 with TensorFlow bilinear resize, float32, add the batch dimension. No crop and no division by 255.

The app handoff recommends center-cropping for a live camera, while training and reference checks resize the whole image. Inspect whether the imported-photo path also crops. For a direct comparison, use the full image without cropping. Check EXIF orientation, rotation, channel conversion, alpha handling, and whether the app uses a thumbnail instead of the selected original.

## Recorded reference predictions and artifact identity

These are prior checks recorded in `docs/MODEL_CONTRACT.md`, not newly executed inference results from this investigation.

| Image | Keras | Default TFLite | Optimized TFLite | Label |
|---|---:|---:|---:|---|
| `dog.png` | 0.99254 | 0.99254 | 0.99545 | dog |
| `cat.jpg` | 0.00225 | 0.00225 | 0.00266 | cat |

The current ML root model files and release copies were hash-checked in this conversation and match the contract:

| Model | SHA-256 |
|---|---|
| `cat_dog_mobilenetv3.tflite` | `cc4ae83dd5c2466a9143e012f4d5b94f8502064ab3b31a0da21b2eb8c94f0c18` |
| `cat_dog_mobilenetv3_optimized.tflite` | `d11d65ea068c1b537ac936cc6ef67e0be9122dd8e266284dbd8f704c11550536` |

The default model has float32 weights and is approximately 3.58 MB. The optimized model uses dynamic-range quantization and is approximately 1.08 MB; its input/output remain float32. No hash check of the installed mobile artifact has been performed.

Reference image hashes from the contract:

- `dog.png`: `a364a3e2ff3bddadde85897aeccb8b3051d9cd6822b79db96fb6d0a4f69b59a1`
- `cat.jpg`: `9b46595fdf92b4a91d380e6b94721da5a68ec18ace40f4829a29db357d79bda5`

## Investigation sequence

1. Locate the actual imported-photo flow, model asset loader, preprocessing, interpreter invocation, output decoding, and UI state update. Inspect the relevant platform-specific implementation as well as shared Kotlin code. Cite file paths and line numbers for findings.
2. Establish which model file the installed app loads. Compare its checksum with the release, accounting for build variants, stale installations, cached assets, and alternate platform models. A matching filename alone is insufficient.
3. Trace the selected image through decoding and resizing into the input tensor. Check raw value range, RGB order, float32 encoding, tensor layout, byte order, buffer position, crop, rotation, and image identity. Do not claim values were measured if only source code was inspected.
4. Trace the single output score through label selection and confidence formatting. Inspect for reversed labels, argmax on one value, double sigmoid, incorrect buffer reads, and a stale prediction displayed for a different image or asynchronous request.
5. Use existing diagnostics or test facilities, if available without writing code, to compare the same reference image. Begin with the default TFLite model on CPU when existing controls permit it, then isolate optimized-model or delegate differences. Do not change app settings or source just to fabricate a completed test; document any measurement that remains unavailable.
6. For the default model and reference preprocessing, the recorded app acceptance rule is matching labels and scores within 0.02 of the reference. Two passing images check basic integration; they do not establish full model accuracy.

## How to interpret the evidence

- If the mobile raw score is above 0.5 but the UI says cat, investigate output mapping and UI state first.
- If the mobile raw score is below 0.5 while Keras/default TFLite gives above 0.5 on the identical image, compare model identity and input tensor values before blaming training.
- If model bytes and input tensor values match but CPU outputs diverge materially, investigate runtime execution and output decoding. Compare delegates separately.
- If both environments classify the identical image as cat, this may be a model error despite good aggregate metrics; the reported discrepancy needs correction.
- Matching Keras/TFLite reference outputs make a general conversion failure less likely. They do not rule out errors on other images or device-specific behavior.

## Required response to the user

Lead with the confirmed cause, or state explicitly that it is not yet confirmed. Separate observed facts, likely explanations, and untested possibilities. Include the affected source locations, the expected versus actual model/input/output behavior, and the smallest proposed correction. State what was actually inspected or run and what remains unknown. Do not implement a fix during this inspection-only task.
