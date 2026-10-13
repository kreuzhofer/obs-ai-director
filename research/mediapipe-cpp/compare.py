"""Compare the ONNX re-implementation against MediaPipe Tasks (Python) on portrait.jpg and warped variants.

usage: python compare.py <models dir: face_landmarker.task, *.onnx, *.tflite, meta.pbtxt> <portrait.jpg>
"""
import math
import pathlib
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import fl_onnx as fl

D = pathlib.Path(sys.argv[1])
IMG = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2RGB)
canon, weights = fl.load_geometry(D / "meta.pbtxt")

tasks = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(D / "face_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_faces=1,
    output_face_blendshapes=True, output_facial_transformation_matrixes=True))

det_s = fl.session(D / "face_detector.onnx")
det_f = fl.session(D / "blaze_face_full_range.onnx")
lmk = fl.session(D / "face_landmarks_detector_b1.onnx")
bls = fl.session(D / "face_blendshapes.onnx")
lmk_cml = fl.session(D / "face_landmarks_detector_cml.onnx", "coreml")


def run_tasks(img):
    r = tasks.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not r.face_landmarks:
        return None
    lm = np.array([(p.x, p.y, p.z) for p in r.face_landmarks[0]])
    return lm, np.array(r.facial_transformation_matrixes[0]), np.array([c.score for c in r.face_blendshapes[0]])


def run_onnx(img, det, kind, lsess=lmk):
    d = fl.detect(det, kind, img)
    if not d:
        return None
    lm, pres = fl.landmarks(lsess, img, fl.roi_from_detection(d[0]))
    return lm, pres, d[0]


def warp(img, yaw=0.0, pitch=0.0, roll=0.0, flip=False):
    """Perspective warp simulating an out-of-plane turn of the (flat) photo."""
    h, w = img.shape[:2]
    if flip:
        img = img[:, ::-1].copy()
    # rotate the photo plane about its own centre, view it from distance f with focal f
    f = 1.2 * w
    R = cv2.Rodrigues(np.array([0, 0, math.radians(roll)]))[0] @ \
        cv2.Rodrigues(np.array([math.radians(pitch), math.radians(yaw), 0.0]))[0]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    P = np.column_stack([src - [w / 2, h / 2], np.zeros(4)]) @ R.T + [0, 0, f]
    dst = np.float32(P[:, :2] / P[:, 2:] * f + [w / 2, h / 2])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


VARIANTS = [("original", {}), ("mirrored", {"flip": True}),
            ("warp yaw +30", {"yaw": 30}), ("warp yaw -30", {"yaw": -30}),
            ("warp yaw +50", {"yaw": 50}), ("warp pitch +25", {"pitch": 25}),
            ("warp pitch -25", {"pitch": -25}), ("roll 30", {"roll": 30}),
            ("roll -60", {"roll": -60}), ("yaw 30 + pitch -20", {"yaw": 30, "pitch": -20})]

print("### Landmark fidelity and head pose (yaw, pitch in degrees)\n")
print("| Input | lmk err ORT-CPU vs Tasks: mean / max px | ORT-CoreML vs ORT-CPU max px | full-range ROI lmk err mean px "
      "| Tasks matrix | Procrustes port on Tasks lmk | Procrustes port on ORT lmk | solvePnP (468 pts, 63 deg) on ORT lmk "
      "| solvePnP on Tasks lmk | blendshape max abs diff |")
print("|---|---|---|---|---|---|---|---|---|---|")
rows = []
for name, kw in VARIANTS:
    img = warp(IMG, **kw) if kw else IMG
    h, w = img.shape[:2]
    t = run_tasks(img)
    o = run_onnx(img, det_s, "short")
    of = run_onnx(img, det_f, "full")
    if t is None or o is None:
        print(f"| {name} | Tasks found face: {t is not None}, ORT found face: {o is not None} |")
        continue
    tl, tm, tb = t
    ol = o[0]
    ol2 = fl.landmarks(lmk, img, fl.roi_from_landmarks(ol * [w, h, 1]))[0]
    err2 = np.linalg.norm((ol2[:, :2] - tl[:, :2]) * [w, h], axis=1)
    px = np.abs((ol[:, :2] - tl[:, :2]) * [w, h])
    err = np.linalg.norm((ol[:, :2] - tl[:, :2]) * [w, h], axis=1)
    cml = fl.landmarks(lmk_cml, img, fl.roi_from_detection(o[2]))[0]
    cml_err = np.linalg.norm((cml[:, :2] - ol[:, :2]) * [w, h], axis=1).max()
    ferr = np.linalg.norm((of[0][:, :2] - tl[:, :2]) * [w, h], axis=1).mean() if of else float("nan")
    yp = lambda R: "%+.1f, %+.1f" % fl.yaw_pitch(R)
    proc_t = fl.mp_pose_matrix(tl, w, h, canon, weights)
    proc_o = fl.mp_pose_matrix(ol, w, h, canon, weights)
    pnp_o = fl.pnp_pose(ol, w, h, canon)[0]
    pnp_t = fl.pnp_pose(tl, w, h, canon)[0]
    ob = bls.run(None, {bls.get_inputs()[0].name: fl.blend_input(ol, w, h)})[0].ravel()
    bd = np.abs(ob - tb).max()
    print(f"| {name} | {err.mean():.2f} / {err.max():.2f} (tracked ROI: {err2.mean():.2f}) | {cml_err:.2f} | {ferr:.2f} | {yp(tm)} | {yp(proc_t)} | "
          f"{yp(proc_o)} | {yp(pnp_o)} | {yp(pnp_t)} | {bd:.3f} |")
    rows.append((fl.yaw_pitch(tm), fl.yaw_pitch(proc_t), fl.yaw_pitch(proc_o), fl.yaw_pitch(pnp_o), fl.yaw_pitch(pnp_t)))

r = np.array(rows)
ref = r[:, 0]
print("\nMax |delta| vs Tasks matrix over all inputs (yaw, pitch):")
for i, n in enumerate(["Procrustes port on Tasks lmk", "Procrustes port on ORT lmk", "solvePnP on ORT lmk",
                       "solvePnP on Tasks lmk"], 1):
    print(f"  {n}: {np.abs(r[:, i] - ref).max(0).round(2)}  mean {np.abs(r[:, i] - ref).mean(0).round(2)}")

# ---- solvePnP variants on the original image
print("\n### solvePnP variants on original portrait (ORT landmarks)")
img = IMG; h, w = img.shape[:2]
tl, tm, _ = run_tasks(img)
ol = run_onnx(img, det_s, "short")[0]
print("Tasks matrix yaw/pitch: %+.2f %+.2f" % fl.yaw_pitch(tm))
basis = np.nonzero(weights)[0]
for label, kw in [("468 pts SQPNP+LM, 63deg", {}), ("32 Procrustes-basis pts", {"idx": basis}),
                  ("468 pts ITERATIVE", {"method": cv2.SOLVEPNP_ITERATIVE}),
                  ("468 pts EPNP no refine", {"method": cv2.SOLVEPNP_EPNP, "refine": False}),
                  ("vFOV 45deg", {"vfov": 45}), ("vFOV 90deg", {"vfov": 90})]:
    print(f"  {label}: yaw/pitch %+.2f %+.2f" % fl.yaw_pitch(fl.pnp_pose(ol, w, h, canon, **kw)[0]))
