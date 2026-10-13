"""Latency per model and per tracked frame on this machine.  usage: python bench.py <models dir> <portrait.jpg>"""
import pathlib
import platform
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from mediapipe.tasks.python import BaseOptions, vision

import fl_onnx as fl

D = pathlib.Path(sys.argv[1])
IMG = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2RGB)
H0, W0 = IMG.shape[:2]
N = 300


def timeit(fn, n=N, warm=30):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return f"{np.median(ts):.2f} (p95 {np.percentile(ts, 95):.2f})"


print(f"ORT {ort.__version__}, TF {tf.__version__}, mediapipe {mp.__version__}, {platform.platform()}\n")
det = fl.detect(fl.session(D / "face_detector.onnx"), "short", IMG)[0]
patch, _ = fl.crop(IMG, fl.roi_from_detection(det))
lm0 = fl.landmarks(fl.session(D / "face_landmarks_detector_b1.onnx"), IMG, fl.roi_from_detection(det))[0]
models = [  # name, onnx (cpu), onnx (coreml), tflite, input
    ("BlazeFace short-range 128", "face_detector", "face_detector_cml", "face_detector.tflite", fl.letterbox(IMG, 128)[0]),
    ("BlazeFace full-range 192", "blaze_face_full_range", "blaze_face_full_range_cml", "blaze_face_full_range.tflite", fl.letterbox(IMG, 192)[0]),
    ("FaceMesh-V2 256", "face_landmarks_detector_b1", "face_landmarks_detector_cml", "face_landmarks_detector.tflite", (patch.astype(np.float32) / 255)[None]),
    ("Blendshapes", "face_blendshapes", "face_blendshapes_cml", "face_blendshapes.tflite", fl.blend_input(lm0, W0, H0)),
]
cols = ["ORT CPU (default threads)", "ORT CPU 1 thread", "CoreML MLProgram ALL", "CoreML MLProgram CPUAndNeuralEngine",
        "CoreML MLProgram CPUAndGPU", "CoreML NeuralNetwork ALL", "TFLite XNNPACK (default)", "TFLite XNNPACK 1 thread"]
print("Median ms per inference (p95), %d runs\n" % N)
print("| Model | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for name, cpu, cml, tfl, x in models:
    row = []
    for s in [fl.session(D / f"{cpu}.onnx"), fl.session(D / f"{cpu}.onnx", threads=1),
              fl.session(D / f"{cml}.onnx", "coreml", "ALL"), fl.session(D / f"{cml}.onnx", "coreml", "CPUAndNeuralEngine"),
              fl.session(D / f"{cml}.onnx", "coreml", "CPUAndGPU"), fl.session(D / f"{cml}.onnx", "coreml", "ALL", "NeuralNetwork")]:
        feed = {s.get_inputs()[0].name: x}
        row.append(timeit(lambda: s.run(None, feed)))
    for th in [None, 1]:
        it = tf.lite.Interpreter(model_path=str(D / tfl), num_threads=th)
        it.allocate_tensors()
        idx = it.get_input_details()[0]["index"]

        def run():
            it.set_tensor(idx, x); it.invoke()
        row.append(timeit(run))
    print(f"| {name} | " + " | ".join(row) + " |")

# ---- per-frame cost in tracking mode (what the plugin does on most frames)
canon, weights = fl.load_geometry(D / "meta.pbtxt")
print("\nPer tracked frame (warpAffine crop + FaceMesh + landmark ROI + Procrustes pose), 1280x720 input:")
frame = cv2.resize(IMG, (int(W0 * 720 / H0), 720))
frame = cv2.copyMakeBorder(frame, 0, 0, (1280 - frame.shape[1]) // 2, 1280 - frame.shape[1] - (1280 - frame.shape[1]) // 2, cv2.BORDER_CONSTANT)
roi = fl.roi_from_detection(fl.detect(fl.session(D / "face_detector.onnx"), "short", frame)[0])
for label, s in [("ORT CPU", fl.session(D / "face_landmarks_detector_b1.onnx")),
                 ("ORT CoreML ALL", fl.session(D / "face_landmarks_detector_cml.onnx", "coreml", "ALL"))]:
    state = {"roi": roi}

    def step():
        lm, _ = fl.landmarks(s, frame, state["roi"])
        state["roi"] = fl.roi_from_landmarks(lm * [1280, 720, 1])
        fl.mp_pose_matrix(lm, 1280, 720, canon, weights)
    print(f"  {label}: {timeit(step)} ms")
lm, _ = fl.landmarks(fl.session(D / "face_landmarks_detector_b1.onnx"), frame, roi)
print(f"  Procrustes port alone (numpy): {timeit(lambda: fl.mp_pose_matrix(lm, 1280, 720, canon, weights))} ms")
print(f"  solvePnP alone (SQPNP+LM, 468 pts): {timeit(lambda: fl.pnp_pose(lm, 1280, 720, canon))} ms")

tasks = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(D / "face_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO, num_faces=1, output_facial_transformation_matrixes=True))
mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
ts = {"t": 0}


def tstep():
    ts["t"] += 33
    tasks.detect_for_video(mpimg, ts["t"])
print(f"  MediaPipe Tasks VIDEO mode (Python, matrix on, blendshapes off): {timeit(tstep)} ms")
