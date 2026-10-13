"""(a) raw conversion fidelity TFLite vs ORT, (b) blendshapes on identical landmarks,
(c) synthetic ground truth for the two head-pose solvers, (d) small/distant faces short- vs full-range.

usage: python fidelity.py <models dir> <portrait.jpg>
"""
import math
import pathlib
import sys

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python import BaseOptions, vision

import fl_onnx as fl

D = pathlib.Path(sys.argv[1])
IMG = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2RGB)
H0, W0 = IMG.shape[:2]
canon, weights = fl.load_geometry(D / "meta.pbtxt")
rng = np.random.default_rng(0)

# ------------------------------------------------------------ (a) raw model fidelity
print("## (a) Raw model outputs, identical input tensors (max abs diff vs TFLite interpreter)\n")
det = fl.detect(fl.session(D / "face_detector.onnx"), "short", IMG)[0]
patch, _ = fl.crop(IMG, fl.roi_from_detection(det))
inputs = {
    "face_detector": fl.letterbox(IMG, 128)[0],
    "blaze_face_full_range": fl.letterbox(IMG, 192)[0],
    "face_landmarks_detector": (patch.astype(np.float32) / 255)[None],
}
tl = fl.landmarks(fl.session(D / "face_landmarks_detector_b1.onnx"), IMG, fl.roi_from_detection(det))[0]
inputs["face_blendshapes"] = fl.blend_input(tl, W0, H0)

print("| Model | output | TFLite range | ORT CPU (tf2onnx) | ORT CPU (CoreML-rewritten graph) | CoreML EP ALL | CoreML EP CPUAndNeuralEngine | CoreML EP CPUAndGPU |")
print("|---|---|---|---|---|---|---|---|")
tfl_name = {"face_detector": "face_detector.tflite", "blaze_face_full_range": "blaze_face_full_range.tflite",
            "face_landmarks_detector": "face_landmarks_detector.tflite", "face_blendshapes": "face_blendshapes.tflite"}
onnx_name = {"face_detector": "face_detector", "blaze_face_full_range": "blaze_face_full_range",
             "face_landmarks_detector": "face_landmarks_detector_b1", "face_blendshapes": "face_blendshapes"}
cml_name = {"face_landmarks_detector": "face_landmarks_detector_cml"}
for m, x in inputs.items():
    it = tf.lite.Interpreter(model_path=str(D / tfl_name[m]))
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    ref = {tuple(o["shape"]): it.get_tensor(o["index"]) for o in it.get_output_details()}
    big = max(ref, key=lambda k: np.prod(k))       # largest output: regressors / landmarks / blendshapes
    r = ref[big].ravel()

    def diff(sess):
        out = sess.run(None, {sess.get_inputs()[0].name: x})
        o = next(v for v in out if v.size == r.size).ravel()
        return f"{np.abs(o - r).max():.2e}"

    cname = cml_name.get(m, onnx_name[m] + "_cml")
    cells = [diff(fl.session(D / f"{onnx_name[m]}.onnx")), diff(fl.session(D / f"{cname}.onnx"))]
    for units in ["ALL", "CPUAndNeuralEngine", "CPUAndGPU"]:
        cells.append(diff(fl.session(D / f"{cname}.onnx", "coreml", units)))
    print(f"| {m} | {big} | {r.min():.1f} .. {r.max():.1f} | " + " | ".join(cells) + " |")

# ------------------------------------------------------------ (b) blendshapes from identical landmarks
tasks = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(D / "face_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_faces=1,
    output_face_blendshapes=True, output_facial_transformation_matrixes=True))
res = tasks.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(IMG)))
tlm = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]])
tbs = np.array([c.score for c in res.face_blendshapes[0]])
bs = fl.session(D / "face_blendshapes.onnx")
ob = bs.run(None, {bs.get_inputs()[0].name: fl.blend_input(tlm, W0, H0)})[0].ravel()
print(f"\n## (b) Blendshapes from the *Tasks* landmarks through ORT: max abs diff vs Tasks = {np.abs(ob - tbs).max():.2e}")

# ------------------------------------------------------------ (c) synthetic ground truth
print("\n## (c) Synthetic ground truth: canonical mesh posed and projected (1280x720, 63 deg vFOV, 60 cm)\n")
W, H = 1280, 720


def synth(yaw, pitch, roll, vfov_true=63.0, dist=60.0, shape_noise=0.0, z_from_model=True):
    R = cv2.Rodrigues(np.array([0, 0, math.radians(roll)]))[0] @ \
        cv2.Rodrigues(np.array([0, math.radians(yaw), 0]))[0] @ cv2.Rodrigues(np.array([math.radians(pitch), 0, 0]))[0]
    pts = canon + rng.normal(0, shape_noise, canon.shape) if shape_noise else canon
    cam = pts @ R.T + [0, 0, -dist]          # MediaPipe metric frame: camera looks down -z
    f = (H / 2) / math.tan(math.radians(vfov_true) / 2)
    u = W / 2 + f * cam[:, 0] / -cam[:, 2]
    v = H / 2 - f * cam[:, 1] / -cam[:, 2]
    # screen z like the landmark model: relative depth, same scale as x (normalised by width)
    z = -(cam[:, 2] - cam[:, 2].mean()) * (f / dist) / W
    lm = np.column_stack([u / W, v / H, z])
    return lm, R


print("| Case | GT yaw, pitch | Procrustes port | solvePnP |")
print("|---|---|---|---|")
worst = {"proc": 0, "pnp": 0}
for (yaw, pitch, roll) in [(0, 0, 0), (30, 0, 0), (-60, 10, 0), (0, -25, 0), (20, 15, 30), (45, -20, -15)]:
    lm, R = synth(yaw, pitch, roll)
    gt = fl.yaw_pitch(R)
    p = fl.yaw_pitch(fl.mp_pose_matrix(lm, W, H, canon, weights)[:3, :3])
    q = fl.yaw_pitch(fl.pnp_pose(lm, W, H, canon)[0])
    print(f"| y{yaw} p{pitch} r{roll} | {gt[0]:+.1f}, {gt[1]:+.1f} | {p[0]:+.2f}, {p[1]:+.2f} | {q[0]:+.2f}, {q[1]:+.2f} |")
for label, kw in [("true vFOV 90 deg, assumed 63", {"vfov_true": 90}), ("true vFOV 40 deg, assumed 63", {"vfov_true": 40}),
                  ("distance 200 cm", {"dist": 200}), ("shape noise 0.3 cm/vertex", {"shape_noise": 0.3})]:
    lm, R = synth(30, -15, 0, **kw)
    gt = fl.yaw_pitch(R)
    p = fl.yaw_pitch(fl.mp_pose_matrix(lm, W, H, canon, weights)[:3, :3])
    q = fl.yaw_pitch(fl.pnp_pose(lm, W, H, canon)[0])
    print(f"| y30 p-15, {label} | {gt[0]:+.1f}, {gt[1]:+.1f} | {p[0]:+.2f}, {p[1]:+.2f} | {q[0]:+.2f}, {q[1]:+.2f} |")

# ------------------------------------------------------------ (d) distant Presenter: small faces in a 1080p frame
print("\n## (d) Small faces: portrait scaled into a 1920x1080 grey frame, face width in px (detector box)\n")
ds, df = fl.session(D / "face_detector.onnx"), fl.session(D / "blaze_face_full_range.onnx")
lmk = fl.session(D / "face_landmarks_detector_b1.onnx")
ref_face_w = det["box"][2]
ref_pose = fl.yaw_pitch(fl.mp_pose_matrix(tl, W0, H0, canon, weights)[:3, :3])
print("| Face width px | Tasks (short-range) found | ORT short-range score | ORT full-range score | yaw, pitch via full-range ROI -> after 1 landmark-ROI pass (ref %+.1f, %+.1f) | yaw, pitch via 4x-face-width crop around Presenter + short-range + 1 landmark-ROI pass |" % ref_pose)
print("|---|---|---|---|---|---|")
for fw in [400, 200, 120, 80, 60, 40, 30, 20]:
    s = fw / ref_face_w
    small = cv2.resize(IMG, (max(1, int(W0 * s)), max(1, int(H0 * s))), interpolation=cv2.INTER_AREA)
    frame = np.full((1080, 1920, 3), 110, np.uint8)
    y0, x0 = 1080 - small.shape[0] - 20 if small.shape[0] < 1060 else 0, 1200
    hh, ww = min(small.shape[0], 1080 - y0), min(small.shape[1], 1920 - x0)
    frame[y0:y0 + hh, x0:x0 + ww] = small[:hh, :ww]
    r = tasks.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    a, b = fl.detect(ds, "short", frame), fl.detect(df, "full", frame)
    pose = "-"
    if b:
        lmf = fl.landmarks(lmk, frame, fl.roi_from_detection(b[0]))[0]
        pose = "%+.1f, %+.1f" % fl.yaw_pitch(fl.mp_pose_matrix(lmf, 1920, 1080, canon, weights)[:3, :3])
        lmf = fl.landmarks(lmk, frame, fl.roi_from_landmarks(lmf * [1920, 1080, 1]))[0]
        pose += " -> %+.1f, %+.1f" % fl.yaw_pitch(fl.mp_pose_matrix(lmf, 1920, 1080, canon, weights)[:3, :3])
    # "Presenter crop": a region 4x the face width around the face, upscaled, short-range detector
    cx, cy = x0 + ww / 2, y0 + hh * 0.35
    side = int(min(1080, max(4 * fw, 64)))
    cx0, cy0 = int(np.clip(cx - side / 2, 0, 1920 - side)), int(np.clip(cy - side / 2, 0, 1080 - side))
    cropi = frame[cy0:cy0 + side, cx0:cx0 + side]
    c = fl.detect(ds, "short", cropi)
    pose_c = "-"
    if c:
        lmc = fl.landmarks(lmk, cropi, fl.roi_from_detection(c[0]))[0]
        lmc = fl.landmarks(lmk, cropi, fl.roi_from_landmarks(lmc * [side, side, 1]))[0]
        pose_c = "%+.1f, %+.1f" % fl.yaw_pitch(fl.mp_pose_matrix(lmc, side, side, canon, weights)[:3, :3])
    print(f"| {fw} | {bool(r.face_landmarks)} | {a[0]['score'] if a else 0:.2f} | {b[0]['score'] if b else 0:.2f} | {pose} | {pose_c} |")
