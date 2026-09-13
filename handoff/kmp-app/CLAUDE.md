# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

A **Kotlin Multiplatform (KMP)** mobile app for Android and iOS that uses the camera to classify, in real time, whether it sees a **cat** or a **dog**.

- **UI is shared:** Compose Multiplatform.
- **Camera and inference are native on each platform:**
  - Android: CameraX + LiteRT.
  - iOS: AVFoundation + TensorFlow Lite or Core ML.
- **The model comes from a separate ML project:** `C:\Users\Usuario\Documents\ai\classification\cat_dogs` (TensorFlow/Keras, MobileNetV3Small transfer learning). Don't retrain or re-export the model here; ask for a new export in that project.

**Status (2026-09-13):** not created yet. The first task is to generate the KMP project skeleton with the versions below.

**Scope rule:** work only inside this app's folder.
- **One exception:** you may **read, never write**, `C:\Users\Usuario\Documents\ai\classification\cat_dogs\release\`, to pick up model releases.
- Don't read or modify anything else in the ML project.

**Read before starting:**
- [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md): exact model input/output, preprocessing, and reference outputs to test against.
- [docs/RESEARCH.md](docs/RESEARCH.md): verified library versions, iOS runtime options, and sources.

## Host environment

- **Windows 11.** Android can be built and run here. **iOS cannot:** it needs macOS with Xcode ≥ 26.4.
- **Android Studio 2026.1.3** is installed, with bundled JBR 25.0.2.
- **No JDK on PATH.** For command-line Gradle, set `JAVA_HOME` to `C:\Program Files\Android\Android Studio\jbr`.
- **Android SDK:** `%LOCALAPPDATA%\Android\Sdk`, platforms 33–36.1, build-tools 36.0.0, emulator image android-36 x86_64 (Play Store).

## Versions (verified 2026-09-13)

| Component | Version | Why |
|---|---|---|
| Kotlin | 2.4.20 | Latest stable (2026-09-07) |
| Compose Multiplatform | 1.12.0 | Current stable |
| Android Gradle Plugin | 9.3.1 | Highest AGP fully supported by Kotlin 2.4.20 (9.4.0 exists, but needs Gradle ≥ 9.6 and may show warnings) |
| Gradle | 9.7.0 | Highest fully supported by Kotlin 2.4.20; runs on JDK 25 (Gradle ≥ 9.1) |
| JDK | 17+ | Android Studio's JBR 25 works |
| compileSdk / build-tools | 36 / 36.0.0 | Already installed |
| minSdk | 23 | Required by LiteRT 2.2.0 `CompiledModel` |
| iOS deployment target | 15.0 | Minimum for Kotlin 2.4 |
| Xcode | ≥ 26.4 | Minimum for Kotlin 2.4 |

## Architecture

**Required with AGP 9:**
- The shared module uses `com.android.kotlin.multiplatform.library` with `kotlin { androidLibrary { } }`.
- The Android app is a separate module (`androidApp`) using `com.android.application`. AGP 9 has Kotlin built in.
- `iosApp` is the Xcode project that consumes the shared framework.

**Platform split:**

| Layer | commonMain | androidMain | iosMain |
|---|---|---|---|
| UI | Compose screens, result label, confidence | `AndroidView` hosting the CameraX `PreviewView` | `UIKitView` hosting an `AVCaptureVideoPreviewLayer` |
| Frames | Frame model (128×128 RGB float32) | CameraX `ImageAnalysis`, `STRATEGY_KEEP_ONLY_LATEST`, `OUTPUT_IMAGE_FORMAT_RGBA_8888` | `AVCaptureVideoDataOutput`, `kCVPixelFormatType_32BGRA`, `alwaysDiscardsLateVideoFrames = true` |
| Inference | `CatDogClassifier` interface, threshold, smoothing | LiteRT 2.2.0 `CompiledModel` | TensorFlowLiteObjC 2.17.0 (pod) or Core ML (see RESEARCH.md) |
| Model file | `composeResources/files/cat_dog_mobilenetv3.tflite` | loaded from resources | loaded from resources, or an `.mlpackage` for Core ML |

- **Kotlin/Native includes Apple framework bindings** (`platform.AVFoundation`, `platform.CoreVideo`, `platform.Accelerate`, `platform.CoreML`, `platform.Vision`), so no dependency is needed for camera or Core ML on iOS.

## Real-time pipeline rules

1. Keep only the newest frame; never queue frames.
2. Apply the sensor rotation, center-crop to a square, and resize to 128×128.
3. Convert to RGB (iOS delivers BGRA) and write **float32 values 0–255, not divided by 255**.
4. The output is one sigmoid value. Above 0.5 is dog, otherwise cat.
5. Smooth the label over the last few frames (for example a moving average of 5) so it doesn't flicker.
6. Run inference off the main thread, and push results to the UI as a `StateFlow`.

## Model updates

- **Current state:** contract **1.0.0**, model release **v1.0.0**. The app's copy of the contract is [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md). Never edit it here; changes come only from the ML project.
- **Releases** are published in `C:\Users\Usuario\Documents\ai\classification\cat_dogs\release\vX.Y.Z\`, each with model files, reference images, `MODEL_CONTRACT.md` and `RELEASE_NOTES.md`.

When the user announces a new release:
1. Read `release/vX.Y.Z/RELEASE_NOTES.md` and `MODEL_CONTRACT.md` from that folder.
2. Act on the release type:

| Type | Version bump | What to do in the app |
|---|---|---|
| **A: weights only** | PATCH | Copy the new model files into `composeResources/files/`, replace `docs/MODEL_CONTRACT.md`, update checksums in tests, run the reference test (±0.02) on Android (and iOS when available) |
| **B: input/output change** | MAJOR | Everything from A, **plus** update preprocessing/postprocessing on Android and iOS (input size, color order, value range, output meaning or threshold) and update tests |
| **C: runtime change** | MINOR | Everything from A, **plus** re-verify on real devices: LiteRT operators and GPU on Android, the iOS runtime (Core ML conversion or TFLite pod), and latency. Update [docs/RESEARCH.md](docs/RESEARCH.md) |

3. Record the synced contract version and release in this file's "Current state" line.
4. If the app can't support a release (for example, an operator isn't supported on iOS), stop and report it to the user. Don't work around it by changing the model contract.

## Conventions

- Write all code in English, including identifiers and user-facing strings.
- Do not add comments in code.
- Preprocessing must follow [docs/MODEL_CONTRACT.md](docs/MODEL_CONTRACT.md) exactly. When touching the classifier, test against its reference outputs.
- Never commit secrets (API keys, signing keys, `local.properties`).
- Put new dependencies in `gradle/libs.versions.toml`.

## Pending checks

These were started in the ML project but not finished:
- Inspect the LiteRT 2.2.0 AAR: native libraries for `arm64-v8a` and `x86_64`, and whether `CompiledModel` is present.
- Run the `.tflite` with LiteRT on a real Android device or emulator and compare with the reference outputs.
- Try converting the model to Core ML (`coremltools` 9.0). If it fails, use the TensorFlowLiteObjC pod on iOS. The conversion itself belongs in the ML project.
- Whether `CompiledModel` GPU needs `<uses-native-library android:name="libOpenCL.so" android:required="false"/>` in the manifest. The official docs don't say.
