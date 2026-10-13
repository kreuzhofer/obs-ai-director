# Running MediaPipe Face Landmarker from a C++ OBS plugin

Resolves [#15](https://github.com/kreuzhofer/obs-ai-director/issues/15) (part of map [#1](https://github.com/kreuzhofer/obs-ai-director/issues/1)). Researched 2026-10-13.

Claims are marked **(measured)** or **(sourced)**.
- **Measured** claims come from the scripts in [`research/mediapipe-cpp/`](mediapipe-cpp/), run on the reference Mac: Apple M2 Max, macOS 26.6, Python 3.12, `onnxruntime` 1.31.0, `mediapipe` 1.1.0, `tensorflow` 2.21.0, `tf2onnx` 1.17.0, `opencv-python` 5.0.
- The test image is Google's `portrait.jpg` (820x1024), plus derived variants (mirrored, perspective-warped, rolled, pasted small into a 1080p frame).
- Models are `face_landmarker.task` (float16, `latest`) and `blaze_face_full_range.tflite` (float16, `latest`) from `storage.googleapis.com/mediapipe-models`.
- Windows was not measured.

## Recommendation

**Convert the bundled `.tflite` models to ONNX and run them on the ONNX Runtime the plugin already uses. Re-implement the small amount of graph glue in C++, and port MediaPipe's geometry pipeline (weighted Procrustes) for head pose instead of using `solvePnP`.**

- **The conversion is lossless.** `tf2onnx` converts all four models (short-range detector, FaceMesh-V2, blendshapes, full-range detector) with no custom ops. Outputs match the TFLite interpreter to about 1e-4 on values that range up to ±200 (measured).
- **The C++ glue is small and fully specified by MediaPipe's source:**
  - SSD anchors, box/keypoint decoding and weighted NMS.
  - A rotated 1.5x ROI and an affine crop to 256x256.
  - Landmark projection, and a landmark-based ROI for the next frame (tracking).
  - About 100 lines of Eigen for the geometry pipeline.

  A numpy re-implementation (`fl_onnx.py`, about 300 lines including comments) gives:
  - landmarks within about 1 px mean of the Tasks API on an 820x1024 image;
  - head yaw/pitch within **1.3 deg mean** (worst 5.4 deg on an extreme warp) of the Tasks `facial_transformation_matrixes` the prototype used (measured).
- **Use the geometry-pipeline port for head pose, not `solvePnP`.** Fed the same landmarks, the port reproduces the Tasks matrix exactly (0.00 deg difference). `cv2.solvePnP` against the canonical mesh is correct on synthetic data, but on real landmarks it has a **13-14 deg mean pitch offset**, and that offset leaks into yaw when the head rolls (measured). It does *not* meet the "within a degree or two" bar.
- **CPU is fast enough on macOS too; CoreML is optional.**
  - Per tracked frame (crop, FaceMesh-V2, ROI update and pose) took **3.4 ms on ORT CPU** (measured, Python loop), against 6.5 ms for MediaPipe Tasks VIDEO mode.
  - The CoreML EP gives no meaningful gain on these tiny models (FaceMesh 2.1-2.7 ms vs 2.9 ms CPU).
  - If CoreML is used, two graph rewrites are mandatory (see below). Without them, ORT 1.31's MLProgram path **fails to compile all four converted models**.
- **Distant Presenters: crop around the Presenter, don't rely on the full-range detector.**
  - In a 1920x1080 frame, the short-range detector (which the Tasks bundle uses) already misses a 200 px-wide face. The full-range detector holds to about 120 px and misses at 80 px (measured).
  - Running the short-range detector on a crop around the Presenter (4x the face width) found the face down to 20 px. With one landmark-ROI pass, pose stayed within about 1.5 deg of the full-resolution reference down to 30 px (measured).
  - The detector only runs to (re)acquire the face. Tracking uses the previous frame's landmarks, so a slower detection path is acceptable.
- **Reject LiteRT and MediaPipe C++/C as the face runtime.**
  - LiteRT would run the `.tflite` files natively. But the conversion is already lossless, LiteRT doesn't remove any glue code, and it would be a second runtime next to ORT.
  - MediaPipe now ships a C Tasks API, and it would give the exact Tasks matrix. But it is only distributed inside the Python wheels as a monolithic 61 MB (Windows) / 129 MB (macOS) shared library. It imports WinINet/CFNetwork, and the only build path is Bazel ("Running MediaPipe on Windows is experimental").

## Comparison

| | **A. `.tflite` → ONNX on ORT** (recommended) | **B. Embed LiteRT / TFLite** | **C. MediaPipe C++/C Tasks** (Bazel source build, or the prebuilt `libmediapipe` from the wheel) |
|---|---|---|---|
| **Conversion fidelity** | tf2onnx 1.17.0, opset 17, no custom ops. Raw-output max abs diff vs TFLite: detector 9e-5, full-range 6e-5, FaceMesh 1.4e-4, blendshapes 9e-7 (measured) | None needed: same `.tflite` files, same kernels (XNNPACK) as Tasks | Exact: it *is* the Tasks graph, including the transformation matrix |
| **Glue code we write** | Anchors/decoding/NMS, ROI and crop, landmark projection, tracking ROI, optional One-Euro smoothing, geometry pipeline. All specified in MediaPipe source (see Evidence) | Same as A: LiteRT runs models, not MediaPipe graphs | None (call `MpFaceLandmarkerDetectForVideo`) |
| **Head pose** | Port the geometry pipeline: exact on identical landmarks; 1.3 deg mean vs Tasks end-to-end (measured) | Same as A | Native `facial_transformation_matrixes` |
| **Licensing** | ORT is MIT, but a static ORT pulls in Apache-2.0 code (onnx, abseil, flatbuffers). Models and canonical mesh are Apache-2.0. The combined binary has to be distributed under GPLv3 terms (allowed by our "or later") | Apache-2.0: same GPLv3 consequence | Apache-2.0: same GPLv3 consequence |
| **Build / static link** | Already solved by #5 (ORT from source, CMake, static) | CMake builds a static `libtensorflow-lite.a` that is "not self-contained" (all transitive deps must be linked too). Prebuilt `libLiteRt.dll/.dylib` exist (shared). A second runtime next to ORT | Bazel only, "Windows is experimental". No static desktop artifact. The wheel's `libmediapipe` is a 61 MB / 129 MB shared library with no supported C distribution (headers from source, unversioned ABI) |
| **Network / telemetry risk** | None if ORT is built with telemetry off (per #5) | Not investigated | Windows DLL imports `WININET.dll` and `WinPixEventRuntime.dll`. The macOS dylib links CFNetwork and contains Clearcut log-source enum strings (no endpoint found; actual network use not verified) |
| **Latency, M2 Max (measured)** | FaceMesh 2.9 ms ORT CPU, 2.1-2.7 ms CoreML. Short detector 0.7 ms; full-range 2.2 ms CPU / 1.7 ms CoreML. **3.4 ms per tracked frame** | FaceMesh 4.4 ms TFLite+XNNPACK (Python interpreter, single thread); about the same as ORT CPU 1-thread (4.8 ms) | Tasks VIDEO mode, Python, matrix on: **6.5 ms per frame** |
| **GPU / NPU** | CoreML EP (after the graph rewrites); DirectML optional per #5 | Metal and WebGPU accelerators | No desktop GPU (per #5) |
| **Verdict** | **Use** | Runner-up; only if a future model won't convert | Reject. Useful as a test oracle (the Python API is this library) |

## Evidence

### What is inside `face_landmarker.task` (measured)

- It is a zip containing `face_detector.tflite` (230 KB), `face_landmarks_detector.tflite` (2.55 MB), `face_blendshapes.tflite` (955 KB) and `geometry_pipeline_metadata_landmarks.binarypb` (19 KB, the canonical face mesh plus Procrustes weights).
- The bundled detector is the **short-range BlazeFace, 128x128 input, 896 anchors**.
- FaceMesh-V2 has a 256x256 input and three outputs: 1434 = 478x3 landmarks, a face-presence logit, and a third tensor.
- The blendshapes model takes 146 landmark (x, y) pairs in image pixels and outputs 52 scores.
- The full-range detector file is **192x192 input with 2304 anchors** (one 48x48 layer).
  - The [Face Detector docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector) list it as 128x128 and the [model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf) says 160x192. The file and [`face_detector_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc) (`kLongRangeImageSize = 192`, `num_boxes 2304`) agree on 192.
  - The model card lists faces "further than 5 meters" as out of scope. License: Apache-2.0 (sourced).
  - `blaze_face_full_range_sparse.tflite` (677 KB) also exists; it was not tested.

### A. Conversion to ONNX

**Conversion (measured).**
- `python -m tf2onnx.convert --tflite X.tflite --opset 17` succeeds for all four models with no custom or contrib ops.
- Op sets:
  - Detectors: Conv, Relu, Add, Pad, MaxPool, Reshape, Transpose, Concat; plus Resize for full-range.
  - FaceMesh: Conv, PRelu (69 nodes), Add, MaxPool, Pad, Sigmoid, Reshape.
  - Blendshapes (MLP-Mixer): Conv, ReduceMean, Mul, Add, Sqrt, Reshape, Transpose, Sigmoid.
- Weights are dequantised from fp16 to fp32, so the ONNX files are about 2x the `.tflite` size: detector 0.42 MB, FaceMesh 4.9 MB, blendshapes 1.9 MB, full-range 2.1 MB.
- FaceMesh comes out with a dynamic batch dimension. Pin it to 1 (`onnxruntime.tools.make_dynamic_shape_fixed`) for CoreML.

**Raw-model fidelity (measured, `fidelity.py` (a)).** Identical input tensors were fed to `tf.lite.Interpreter` and to ORT. Max abs diff on the largest output:

| Model | Output range | ORT CPU | ORT CoreML EP (MLProgram, ALL / CPUAndNeuralEngine / CPUAndGPU) |
|---|---|---|---|
| Short-range detector (regressors) | -54 .. 234 | 9.2e-5 | 1.4e-4 / 9.2e-5 / 1.4e-4 |
| Full-range detector (regressors) | -64 .. 96 | 6.1e-5 | 1.4e-4 / 6.3e-5 / 1.4e-4 |
| FaceMesh-V2 (landmarks, px in the 256 crop) | -38 .. 210 | 1.4e-4 | 1.1e-4 / 1.5e-4 / 1.1e-4 |
| Blendshapes | 0 .. 1 | 8.9e-7 | 5.4e-4 (all three) |

The blendshapes model fed the Tasks API's own landmarks through ORT matches Tasks' blendshapes to 4.8e-7. So any end-to-end difference comes from preprocessing (the ROI and crop), not from the conversion.

**End-to-end vs the Tasks API (measured, `compare.py`).**
- ORT landmarks differ from Tasks IMAGE-mode landmarks by **0.7-1.5 px mean** (max 1.7-6.3 px) on the original, mirrored, warped and rolled variants. The extreme 50 deg warp gave 2.3 px mean.
- The gap comes from the ROI/crop step: MediaPipe's `ImageToTensor` letterboxing and sampling are not replicated bit-exactly. It is at the level of landmark jitter.
- CoreML-EP landmarks equal CPU landmarks to under 0.01 px.

**CoreML EP: two mandatory graph rewrites (measured, `coreml_diag.py`, `coreml_fix.py`).**
1. tf2onnx omits the `pads` attribute on Conv/MaxPool when padding is zero. ORT 1.31's CoreML **MLProgram** builder then emits an invalid program, and session creation fails for **all four models** with `Unable to parse ML Program: ... Required param 'pad' is missing`. NeuralNetwork format compiles. Fix: add explicit `pads=[0,0,0,0]`.
2. The CoreML EP supports neither **PRelu** nor **Neg**. FaceMesh's 69 PRelu nodes split it into 70 CoreML partitions (154/223 nodes on CoreML). Fix: rewrite `PRelu(x, a)` as `Relu(x) + a*(x - Relu(x))`. After that FaceMesh is **1 partition, 430/430 nodes**, and both detectors are 1 partition each. The blendshapes model stays at 9 partitions; it isn't needed for v1.

Both rewrites are numerically exact: on ORT CPU, the rewritten graphs give the same diff against TFLite as the originals (measured). Do them at model-preparation time, not in the plugin.

**What the C++ glue must replicate.** Sourced from MediaPipe `master`; every parameter is mirrored in `fl_onnx.py`:
- **Detector** ([`face_detector_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc)):
  - Input: letterboxed (`keep_aspect_ratio`, `BORDER_ZERO`), RGB in [-1, 1].
  - SSD anchors, short-range: 4 layers, strides 8/16/16/16, `interpolated_scale_aspect_ratio 1.0`, `fixed_anchor_size`.
  - SSD anchors, full-range: 1 layer, stride 4, interpolated ratio 0.
  - Decoding: sigmoid score clipped at ±100, `reverse_output_order`, 6 keypoints.
  - Weighted NMS on IoU.
  - ROI rotation from keypoint 0 to keypoint 1 (the eyes), target angle 0, then the rect is scaled by 1.5.
- **Landmarks** ([`face_landmarks_detector_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc)):
  - Input: the rotated ROI warped to 256x256, RGB in [0, 1]; output is projected back through the ROI.
  - Next-frame ROI from landmarks 33→263 (eye corners), scale 1.5, `square_long`.
  - Smoothing: a One-Euro filter (`min_cutoff 0.05`, `beta 80`, `derivate_cutoff 1`).
- **Blendshapes** ([`face_blendshapes_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc)): a fixed list of 146 landmark indices, (x, y) in image pixels.

### B. LiteRT / TFLite

- LiteRT 2.3.0 (2026-10-07) is Apache-2.0. Its release assets are Apple xcframeworks plus `litert_cc_sdk.zip`. Prebuilt `libLiteRt.dll` (Windows x86_64) and `libLiteRt.dylib` (macOS arm64) are kept in-repo under `litert/prebuilt/` (Git LFS) ([releases](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0), [prebuilt](https://github.com/google-ai-edge/LiteRT/tree/main/litert/prebuilt)) (sourced).
- `litert/CMakeLists.txt` has `LITERT_AUTO_BUILD_TFLITE`, `LITERT_ENABLE_GPU` and `TFLITE_ENABLE_XNNPACK` options (sourced). The [CMake build guide](https://ai.google.dev/edge/litert/build/cmake) notes that the static `libtensorflow-lite.a` "isn't self-contained since all the transitive dependencies are not included" (sourced).
- LiteRT only runs models. MediaPipe's anchors, ROI, tracking and geometry are calculators in the MediaPipe graph, not in the model. So option B needs exactly the same glue as option A, while adding a second runtime next to ORT (which #5 chose to also serve gestures and VAD later).
- Latency is equivalent to ORT CPU (measured; see the table below).

### C. MediaPipe C++ / C Tasks

- MediaPipe now has a **C Tasks API**: [`mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h) exposes `MpFaceLandmarkerCreate`, `MpFaceLandmarkerDetectForVideo` and others with `MP_EXPORT` (sourced).
- **The `mediapipe` 1.x Python package is a ctypes wrapper around a prebuilt `libmediapipe`** (`tasks/python/core/mediapipe_c_bindings.py` loads it with `ctypes.CDLL`) (measured).
  - macOS arm64: `mediapipe/tasks/c/libmediapipe.dylib` is **129 MB**. It exports `_MpFaceLandmarker*`, links Metal, AVFoundation, AppKit, CFNetwork and Security, and contains Clearcut log-source enum strings. No logging endpoint URL was found, and actual network use was not tested.
  - Windows x64 (`mediapipe-1.1.0-py3-none-win_amd64.whl`): `libmediapipe.dll` is **61 MB**. It references `WININET.dll`, `OpenCL.dll` and `WinPixEventRuntime.dll` (the last is not an OS DLL).
- This is technically a fourth path: `dlopen` the wheel's library from the plugin. But it is not a supported C SDK distribution. Headers come from the source tree, there is no ABI/versioning promise, it is a shared library of 60-130 MB, and the network imports conflict with the no-telemetry rule unless audited.
- Building from source is Bazel only. The [install docs](https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md) say "Running MediaPipe on Windows is experimental", and v1.1.0's notes include "Fix Windows build" ([v1.1.0](https://github.com/google-ai-edge/mediapipe/releases/tag/v1.1.0)) (sourced). #5 already rejected MediaPipe as the runtime for these reasons.
- **Latency (measured):** Tasks VIDEO mode via this library, called from Python, 1280x720, matrix on and blendshapes off: **6.49 ms median** (p95 6.77).

### Head pose without the Tasks matrix

**How Tasks computes the matrix (sourced)** ([`geometry_pipeline.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc), [`procrustes_solver.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/libs/procrustes_solver.cc), [`face_geometry_from_landmarks_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/face_geometry_from_landmarks_graph.cc)):
- It uses the first 468 landmarks (iris excluded) and a virtual perspective camera: **vertical FOV 63 deg**, near plane 1, origin top-left.
- XY is projected onto the near plane. The canonical-to-runtime scale is estimated twice with a weighted orthogonal Procrustes solve, the landmarks are unprojected using the landmark z, and a final weighted Procrustes gives the 4x4 transform.
- Weights: 32 non-zero `procrustes_landmark_basis` entries in [`geometry_pipeline_metadata_landmarks.pbtxt`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt), which also holds the canonical mesh.
- Porting it is about 60 lines of numpy (`fl_onnx.mp_pose_matrix`), about 100 lines with Eigen.

**Results (measured):**

| Pose source | vs Tasks matrix: mean / max abs diff (yaw, pitch) over 10 image variants |
|---|---|
| Geometry-pipeline port on the **Tasks** landmarks | **0.00 / 0.00 deg** (exact) |
| Geometry-pipeline port on **ORT** landmarks (full re-implementation) | **1.3, 1.3 / 5.4, 2.6 deg** |
| `cv2.solvePnP` (SQPNP + LM refine, 468 points, 63 deg vFOV) on ORT landmarks | 4.1, 13.8 / 18.5, 18.2 deg |
| `cv2.solvePnP` on the Tasks landmarks | 4.0, 14.0 / 18.0, 17.1 deg |

**Why PnP is off.** On a synthetic canonical face posed and projected with known intrinsics (`fidelity.py` (c)), both solvers are right: PnP is exact and the port is within 0.8 deg. So the PnP offset on a real face is a model-mismatch effect, not a bug:
- PnP fits only the 2D projection of a *generic* face. Pitch is weakly constrained from 2D, so the solver tilts the canonical face to match this person's vertical proportions.
- MediaPipe's pipeline uses the network's per-landmark depth, so it is 3D-to-3D.
- On the frontal portrait, PnP reads pitch -16.5 deg against Tasks' -3.9 deg. The offset depends on the assumed FOV (-11.9 at 45 deg, -23.6 at 90 deg) and on the solver (EPnP without refinement gives -7.5).
- Because the offset sits in the face's own frame, a 30 deg roll turns about 10 deg of it into yaw.
- The 32 Procrustes-basis points instead of all 468 give the same result.

**Off-axis Presenters (measured, `offaxis.py`).**
- The Tasks matrix (and the port) give rotation in the **camera frame**, assuming a 63 deg vFOV camera regardless of the real lens.
- A face pasted at increasing horizontal offsets in a 1080p frame keeps roughly the same camera-frame yaw (-4 to -10 deg). Its yaw relative to the face-to-camera line of sight grows with the bearing: +18 deg at 21 deg off-axis, +34 deg at 44 deg off-axis.
- For a real Presenter looking *into the lens* from the edge of a wide shot, the camera-frame yaw is about minus their bearing (up to about ±40 deg at a 1080p frame edge under the 63 deg vFOV assumption).
- The Attention rule should therefore use yaw relative to the line of sight (from the translation part of the same matrix) or keep Presenters near the frame centre. Either way the real FOV matters; see open questions.

### Full-range detector and distant Presenters (measured, `fidelity.py` (d))

The portrait was scaled down and pasted into a grey 1920x1080 frame:

| Face width (px) | Tasks (short-range) finds face | ORT short-range score | ORT full-range score | Pose via full-range ROI → after one landmark-ROI pass | Pose via short-range on a 4x-face-width crop + one landmark-ROI pass |
|---|---|---|---|---|---|
| 400 | no | 0.65 (just above 0.5) | 0.85 | -8.0, -0.7 → -9.6, -3.4 | n/a (crop larger than frame) |
| 200 | no | none | 0.92 | -5.5, -1.1 → -6.5, -1.7 | +1.5, -1.6 |
| 120 | no | none | 0.81 | -4.2, +3.1 → -4.0, +2.5 | +1.6, -2.1 |
| 80 | no | none | none | n/a | +1.6, -2.1 |
| 60 / 40 / 30 | no | none | none | n/a | +1.5, -1.8 / +1.4, -3.1 / +1.4, -4.1 |
| 20 | no | none | none | n/a | +0.3, -5.6 |

Reference pose at full resolution: +1.5, -2.7.

- The full-frame and crop columns differ in yaw partly because they assume different virtual cameras: the crop is treated as its own centred 63 deg camera. That is the off-axis effect above, not detector quality.
- For the plugin, map crop landmarks back to full-frame coordinates before solving pose (as `offaxis.py` does).
- Practical rule: run the detector on a Presenter crop (or a few tiles) when acquiring the face, then track with landmark ROIs. The short-range detector on a crop (0.7 ms) beats the full-range detector on the whole frame (2.2 ms) and still works on much smaller faces.

### Latency (measured, `bench.py`, median ms over 300 runs, p95 in brackets)

| Model | ORT CPU (default threads) | ORT CPU 1 thread | CoreML MLProgram ALL | CoreML MLProgram CPU+ANE | CoreML MLProgram CPU+GPU | CoreML NeuralNetwork ALL | TFLite XNNPACK |
|---|---|---|---|---|---|---|---|
| BlazeFace short-range 128 | 0.68 (0.77) | 1.38 | 1.37 | 0.75 | 1.39 | 2.29 | 1.18 |
| BlazeFace full-range 192 | 2.18 (2.41) | 4.71 | 1.74 | 2.50 | 1.86 | 3.76 | 4.01 |
| FaceMesh-V2 256 | 2.92 (3.39) | 4.82 | 2.38 (4.07) | 2.66 | 3.39 | 2.10 | 4.40 |
| Blendshapes | 0.59 | 0.93 | 0.93 | 0.94 | 0.94 | 2.04 | 0.99 |

- Per tracked frame on a 1280x720 input (warpAffine crop, FaceMesh, landmark ROI and pose), all in Python: **ORT CPU 3.43 ms** (p95 3.78), ORT CoreML 4.10 ms.
- The pose step alone: Procrustes port 0.19 ms, `solvePnP` 0.41 ms.
- MediaPipe Tasks VIDEO mode: 6.49 ms.
- The CoreML models are fp32 MLProgram, so the Neural Engine is unlikely to be used. An fp16 export might change that; not tested.
- At 15 fps per Camera this is about 5% of one core per Camera on CPU.

### Licensing (sourced)

- The models, canonical mesh and MediaPipe source are Apache-2.0 (model cards; [repo](https://github.com/google-ai-edge/mediapipe)). LiteRT is Apache-2.0. ONNX Runtime is MIT, but a static ORT build includes Apache-2.0 dependencies such as `onnx`, `abseil_cpp` and `flatbuffers` ([`cmake/deps.txt`](https://github.com/microsoft/onnxruntime/blob/main/cmake/deps.txt)).
- The FSF says Apache-2.0 is "compatible with version 3 of the GNU GPL" but "not compatible with GPL version 2" ([license list](https://www.gnu.org/licenses/license-list.html#apache2)).
- OBS Studio is "version 2 of the License, or (at your option) any later version" ([libobs/obs.c](https://github.com/obsproject/obs-studio/blob/master/libobs/obs.c)), as is the plugin template.
- Consequence: with *any* of the three options, the distributed plugin binary is a GPLv3 combination. Our GPL-2.0-or-later source licence allows that, and it is the same position obs-backgroundremoval (GPL-3.0) is in. The choice of face runtime doesn't change it.
- Shipping the models as separate downloads keeps them data, not linked code.

## Open questions

1. **Attention geometry for off-axis Presenters.**
   - The head yaw the prototype used is camera-frame yaw under an assumed 63 deg vFOV. A Presenter looking into the lens from a frame edge reads tens of degrees of yaw.
   - Decide between: a per-Camera FOV setting plus line-of-sight correction, the Director's relative "most frontal Camera wins" rule from #4 (which partly cancels the effect), or both.
   - Measure it with a real Presenter at the frame edge; these tests only used a pasted photo.
2. **Real head-turn validation.** No ground-truth head-pose dataset was reachable (AFLW2000-3D host didn't resolve; BIWI returned 403). The warped-photo variants don't produce large apparent head turns: the Tasks yaw stayed within ±5 deg. Before relying on thresholds, log Tasks matrix vs the ORT port side by side on live webcam frames with real ±60 deg turns (cheap to add to the prototype).
3. **Binary licence statement.** Record in an ADR that release binaries are distributed under GPLv3 terms because of the Apache-2.0 components. This is independent of the face runtime.

## Reproducing

```sh
uv init --python 3.12 env && cd env
uv add tf2onnx tensorflow onnxruntime mediapipe opencv-python numpy onnx
# models dir: face_landmarker.task, blaze_face_full_range.tflite, portrait.jpg (see links above)
uv run sh ../research/mediapipe-cpp/convert.sh <models dir>
cd ../research/mediapipe-cpp
uv run --project ../../env python compare.py  <models dir> <models dir>/portrait.jpg
uv run --project ../../env python fidelity.py <models dir> <models dir>/portrait.jpg
uv run --project ../../env python offaxis.py  <models dir> <models dir>/portrait.jpg
uv run --project ../../env python bench.py    <models dir> <models dir>/portrait.jpg
```

No model weights are committed.
