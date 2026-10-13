# Research: Local models for head pose and gaze

Ticket: [#4](https://github.com/kreuzhofer/obs-ai-director/issues/4) (part of map [#1](https://github.com/kreuzhofer/obs-ai-director/issues/1)).
Question: which locally runnable models best tell whether the Presenter is facing, or looking into, a Camera?

Researched 2026-10-13. Latency figures marked **(measured)** come from a quick benchmark on the reference Mac (Apple M2 Max, macOS, Python 3.14, `mediapipe` 1.1.0, `onnxruntime` 1.31.0). They time steady-state inference on a synthetic or still input and are only meant for comparing orders of magnitude. Windows/RTX figures were not measured.

## Recommendation

1. **Prototype first with MediaPipe Face Landmarker. It is the single dependency for head pose and the first-pass eye signal.**
   - One `.task` bundle (3.7 MB, Apache-2.0) includes the face detector, 478 landmarks (including iris), 52 blendshapes and a 4x4 facial transformation matrix (head rotation and translation).
   - With blendshapes and the matrix enabled it ran in about **7 ms per frame on M2 Max CPU (measured)**. At 15 fps that is about 10% of one core per Camera.
   - Head pose comes from the rotation part of the matrix.
   - Eye-in-head direction comes from the `eyeLook{In,Out,Up,Down}{Left,Right}` blendshapes, or from iris landmarks against eye corners. Combined with head pose, that gives a coarse gaze.
   - It also runs as a C++ graph and on Windows/macOS CPU, which matters for the later OBS plugin.
2. **Add a learned gaze model only if the prototype shows MediaPipe eye signals are too weak to separate "looking into this Camera" from "facing it but looking at notes or a screen".** Start with the `yakhyo/gaze-estimation` L2CS-style ONNX models:
   - MobileOne-S0 is 4.8 MB, with 12.58 deg MAE on Gaze360. It took **10 ms on CPU and 2 ms with the ORT CoreML EP (measured)**.
   - ResNet-50 took **165 ms on CPU and 4 ms with CoreML (measured)**.
   - Feed them the face crop from MediaPipe so no extra detector is needed.
3. **Keep 6DRepNet as a fallback for head pose only**, for when MediaPipe's head pose breaks down at large yaw (the FaceMesh card says it is not suited to faces turned more than 80 deg). It has the best published accuracy here (AFLW2000 MAE 3.97 deg) but is a 39 M-param model:
   - **27 ms** with torch on CPU, **88 ms** with ORT on CPU, and **3.7 ms** with ORT CoreML (all measured).
   - It needs a separate face detector.
4. **Avoid OpenFace 2.x/3.0, UniGaze and ETH-XGaze models.** All are non-commercial or research-only, and none brings anything MediaPipe plus a small gaze CNN lacks for this use.

**Design hint for the Director (not a model choice):** each Camera sees the Presenter from a different angle. So "which Camera is attended" can be decided *relatively*: pick the Camera whose view shows the most frontal head, or the gaze closest to the optical axis (smallest |yaw| and |pitch|). That needs no absolute calibration and is far more forgiving of 5-12 deg model error than a per-camera threshold. Any absolute "looking *into* the lens" test needs eye gaze, and even so 10+ deg in-the-wild error is comparable to the angle between a lens and a nearby monitor.

## Comparison

| Candidate | Head pose | Eye gaze | Published accuracy | Input to net | Latency, M2 Max (measured unless noted) | Formats available | Needs separate face detector | Weight / code license |
|---|---|---|---|---|---|---|---|---|
| **MediaPipe Face Landmarker** (BlazeFace + FaceMesh-V2 + Blendshape V2) | Yes: 4x4 transform matrix (Procrustes against a canonical face) | Indirect: 8 `eyeLook*` blendshapes plus 10 iris landmarks; no gaze angles | Landmark IOD-normalised MAE (no head-pose or gaze angle benchmark published) | Detector 192x192; mesh 256x256 crop; blendshape MLP 146 landmarks | **~7 ms/frame CPU**, full pipeline with blendshapes and matrix (VIDEO mode) | `.task` (TFLite fp16); runtimes for Python, C++, Android, iOS, Web | No (bundled) | Apache-2.0 |
| **L2CS-Net** (original, ResNet-50) | No | Yes: yaw/pitch, 90 bins of 4 deg | MPIIGaze 3.92 deg; Gaze360 10.41 deg | 224x224 face crop in the official pipeline (the LiteRT port uses 448) | not measured (see the yakhyo ResNet-50 row; the LiteRT port's model card reports 11.8 ms WebGPU on M4 Max) | PyTorch `.pkl`; third-party LiteRT `.tflite` | Yes (RetinaFace) | MIT code/weights; **trained on Gaze360, whose data is non-commercial-research only** |
| **yakhyo/gaze-estimation** (L2CS-style, several backbones) | No | Yes: yaw/pitch | Gaze360 MAE: MobileOne-S0 12.58, MobileNetV2 13.07, ResNet-18 12.84, ResNet-34 11.33, ResNet-50 11.34 deg | 448x448 (ONNX graphs are fixed) | MobileOne-S0: **10.3 ms CPU / 2.0 ms CoreML EP**; ResNet-50: **165 ms CPU / 4.1 ms CoreML EP** | `.pt` and **ONNX** (GitHub release) | Yes (RetinaFace via `uniface`), or reuse the MediaPipe crop | MIT; Gaze360-trained (same caveat) |
| **6DRepNet** (RepVGG-B1g2) | Yes: 3x3 rotation matrix | No | AFLW2000 MAE 3.97 deg; BIWI 3.47 deg (trained on 300W-LP) | 224x224 | **27 ms torch CPU; 88 ms ORT CPU (4 threads); 3.7 ms ORT CoreML EP** (exports cleanly to ONNX) | `.pth`; `pip install sixdrepnet`; ONNX via `torch.onnx.export` | Yes (`face_detection` RetinaFace) | MIT; 300W-LP training data |
| **6DRepNet360** | Yes, full 360 deg range | No | Official README publishes no numbers (aggregators list BIWI ~3.4 deg, unverified) | 224x224 (same family) | not measured | `.pth` only; README usage says "Coming soon!" | Yes | MIT; trained on 300W-LP and CMU Panoptic |
| **OpenFace 2.x** | Yes | Yes | n/a | n/a | "real-time" (no numbers) | C++ binaries | built-in | **Non-commercial research only** (CMU) |
| **OpenFace 3.0** | Not listed in 3.0 scope | Yes (per-eye yaw/pitch) | see paper | n/a | n/a | PyTorch | RetinaFace | **Non-commercial research only** (CMU) |
| **UniGaze** (ViT-B/L/H) | No | Yes, strongest cross-domain generalisation claim | see paper | ViT (heavy) | not measured; ViT-H is impractical alongside streaming | PyTorch, `pip install unigaze` | Yes (needs normalisation) | **MG-NC-RAI-2.0 (non-commercial)** |
| **ETH-XGaze baseline** | No | Yes, extreme head-pose range | see paper | 224x224 normalised | not measured | PyTorch | Yes | **CC BY-NC-SA 4.0** |

## Evidence

### MediaPipe Face Landmarker
- Outputs: 478 3D landmarks, 52 blendshape scores (`output_face_blendshapes`) and facial transformation matrices (`output_facial_transformation_matrixes`). Bundle: BlazeFace short-range detector 192x192, FaceMesh-V2 256x256, blendshape model 1x146x2, float16. Running modes: IMAGE, VIDEO, LIVE_STREAM. Source: [Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
- Blendshape names include `eyeLookDownLeft/Right`, `eyeLookInLeft/Right`, `eyeLookOutLeft/Right` and `eyeLookUpLeft/Right`. Source: [`face_blendshapes_graph.cc`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc).
- The transform matrix is a rigid mapping from the canonical metric face to the runtime landmarks, fitted by Procrustes analysis on the CPU through a virtual perspective camera. The docs recommend matching the virtual camera to the real one for better results. Source: [legacy Face Mesh docs, "Face Transform Module"](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md).
- FaceMesh V2 model card (Apache-2.0, 2022-09-15):
  - MobileNetV2-like CNN; input is the face crop with a 25% margin, resized to 256x256.
  - Not suitable for faces "looking away from the camera (more than 80°)", "only partially visible (less than 50% of the face)", or "too far away from the camera (cropped face can't be rescaled to model input of 192x192 without quality degradation)".
  - Expect "jittering" under degraded light, noise or motion.
  - Source: [Model Card MediaPipe Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf).
- Blendshape V2 model card (Apache-2.0): MLP-Mixer on 146 landmarks that outputs 52 coefficients in [0,1]. It is trained on synthetic GHUM-mesh data and is intended for "AR entertainment / avatar control", so its eye-look values are not calibrated gaze angles. Source: [Model Card Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
- Python GPU delegate: the `BaseOptions` docstring says "GPU support is currently limited to Ubuntu platforms". So the Python prototype runs MediaPipe on CPU on both macOS and Windows, which is fine at about 7 ms. Source: [`base_options.py`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py).
- Small faces and distance: the bundled detector is the short-range BlazeFace ("selfie-style, close range"). Full-range and sparse full-range detectors exist as separate Face Detector models (128x128 input, per the docs). The Face Detector short-range model is listed at 2.94 ms CPU on a Pixel 6. Source: [Face Detector guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector). For a Presenter far from a wide Camera, crop a region around the Presenter from the full-resolution frame instead of feeding the whole downscaled frame.
- **Measured (M2 Max CPU, VIDEO mode, blendshapes and matrix on):** 7.07 ms/frame at 640x799 and 7.10 ms/frame at 960x1198 input, on Google's sample `portrait.jpg`. The near-frontal sample gave a rotation close to identity (about 2 deg yaw, 4.5 deg pitch) and all `eyeLook*` scores of 0.03-0.18.

### L2CS-Net
- Paper: angular error 3.92 deg on MPIIGaze and 10.41 deg on Gaze360; each angle is regressed separately with a per-angle loss. Source: [arXiv:2203.03339](https://arxiv.org/abs/2203.03339).
- Repo (MIT): ResNet-50 backbone, weights on Google Drive (`L2CSNet_gaze360.pkl`), PyTorch only. Source: [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net).
- The official pipeline uses `from face_detection import RetinaFace`, `cv2.resize(img, (224, 224))` and 90 output bins. Source: [`l2cs/pipeline.py`](https://github.com/Ahmednull/L2CS-Net/blob/main/l2cs/pipeline.py).
- The third-party LiteRT port (MIT) uses 448x448 input and fp16 `.tflite` (47.9 MB). It reports 11.8 ms with WebGPU on M4 Max and 541.7 ms on Pixel 8a CPU with 4 XNNPACK threads, which shows how CPU-heavy ResNet-50 at 448 is. Source: [litert-community/L2CS-Gaze360-LiteRT](https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT).
- Training data: Gaze360's README says "The usage of the dataset and the code is for non-commercial research use only". **Flag: the weights are MIT-labelled but derived from non-commercial data.** Source: [erkil1452/gaze360](https://github.com/erkil1452/gaze360).

### yakhyo/gaze-estimation
- MIT. Backbones: ResNet-18/34/50, MobileNetV2 and MobileOne-S0, all trained on Gaze360 only. Gaze360 MAE as in the table; ONNX and `.pt` weights are published as GitHub release assets (release "weights", updated 2026-02-14). It uses RetinaFace via `uniface`. Source: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation).
- **Measured (M2 Max, ORT 1.31, 4 intra-op threads):** input `[1,3,448,448]`, outputs two `[1,90]` tensors (yaw and pitch bins).
  - MobileOne-S0: 10.25 ms on CPU EP, 1.96 ms on CoreML EP.
  - ResNet-50: 164.98 ms on CPU EP, 4.14 ms on CoreML EP.
  - On Windows the equivalent is ORT with the CUDA, DirectML or TensorRT EPs (not measured).

### 6DRepNet / 6DRepNet360
- 6DRepNet (MIT): trained on 300W-LP; AFLW2000 MAE 3.97 deg (yaw 3.63, pitch 4.91, roll 3.37); BIWI MAE 3.47 deg; backbone RepVGG-B1g2; `pip install sixdrepnet`; demo uses the `face_detection` RetinaFace package. Source: [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet). Paper: [arXiv:2202.12555](https://arxiv.org/abs/2202.12555).
- Inference preprocessing is `Resize(224)` then `CenterCrop(224)`, and weights download from `cloud.ovgu.de`. Source: [`sixdrepnet/regressor.py`](https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/regressor.py).
- **Gotcha (observed):** `sixdrepnet/utils.py` hard-codes `torch.device('cuda:%d' % gpu)` for any non-CPU tensor, so the pip package fails on Apple MPS. Run it on CPU, or export to ONNX and use CoreML EP. The ONNX export worked with no changes.
- **Measured (M2 Max):** 39.3 M params. Torch CPU took 26.99 ms. The exported ONNX (input `[1,3,224,224]`, output `[1,3,3]` rotation matrix) took 87.99 ms on ORT CPU EP (4 threads) and 3.70 ms on ORT CoreML EP, with 104 of 114 nodes on CoreML.
- 6DRepNet360 (MIT) targets the full rotation range. It is fine-tuned on 300W-LP and on CMU Panoptic for the "Full Rotation" variant, and only `.pth` weights are offered; README usage says "Coming soon!". Sources: [thohemp/6DRepNet360](https://github.com/thohemp/6DRepNet360), paper [arXiv:2309.07654](https://arxiv.org/abs/2309.07654).

### OpenFace
- OpenFace 2.x provides landmarks, head pose, AUs and eye gaze. Its license grants use "for noncommercial research purposes" only, with no distribution, and derivatives are owned by CMU. Sources: [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), [OpenFace-license.txt](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt).
- OpenFace 3.0 provides landmarks, AUs, gaze and emotion, with RetinaFace inside. Its LICENSE is titled "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY". Sources: [CMU-MultiComp-Lab/OpenFace-3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0), paper [arXiv:2506.02891](https://arxiv.org/abs/2506.02891).
- **Flag: non-commercial.** It is also a heavy dependency for a GPL OBS plugin; not recommended.

### UniGaze and ETH-XGaze
- UniGaze has ViT-B/L/H backbones with self-supervised pre-training on large in-the-wild face data, and claims consistently better cross-domain gaze. `pip install unigaze`; weights come from Hugging Face. **License: ModelGo Attribution-NonCommercial-ResponsibleAI 2.0.** Sources: [ut-vision/UniGaze](https://github.com/ut-vision/UniGaze), [arXiv:2502.02307](https://arxiv.org/abs/2502.02307).
- ETH-XGaze is a dataset and baseline for extreme head pose and gaze. The repo code and pre-trained model are under **CC BY-NC-SA 4.0**. Sources: [xucong-zhang/ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze), [arXiv:2007.15837](https://arxiv.org/abs/2007.15837).

### Not verified
- Apple Vision `VNFaceObservation` `yaw`/`pitch`/`roll` is an OS-native, macOS-only head-pose option with no weights to download. Apple's documentation pages could not be fetched in this session, so availability and precision are unverified. It may be worth a look only if the Mac build wants zero model downloads; it is not cross-platform.
- MediaPipe latency on Windows (Ryzen 9 5950X3D) and ORT CUDA/DirectML latency on the RTX 4070 Ti Super were not measured.

## Open questions this raises
- Are MediaPipe's `eyeLook*` blendshapes plus iris geometry good enough to tell "into the lens" from "at the screen next to it"? Only the live-webcam prototype can answer this. It decides whether a second (gaze) model is needed at all.
- Sizing the face crop for distant Presenters: cropping a full-resolution region around the last known face, versus a full-range detector.
- Training-data licenses: L2CS-family and 6DRepNet weights are MIT-labelled but trained on research-only datasets (Gaze360; 300W-LP derives from 300W). This is informational per the map, but worth stating in the model-download UX.
