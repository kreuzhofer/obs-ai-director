"""Same 200 px face pasted at different horizontal positions of a 1920x1080 frame.
Camera-frame yaw (what the Tasks matrix gives) vs yaw relative to the line of sight face->camera.
Landmarks via 'Presenter crop' are mapped back to full-frame coordinates before solving pose.

usage: python offaxis.py <models dir> <portrait.jpg>
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
ds, df = fl.session(D / "face_detector.onnx"), fl.session(D / "blaze_face_full_range.onnx")
lmk = fl.session(D / "face_landmarks_detector_b1.onnx")
W, H = 1920, 1080
tasks = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(D / "face_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_faces=1, output_facial_transformation_matrixes=True))

ref_w = fl.detect(ds, "short", IMG)[0]["box"][2]


def los_yaw_pitch(T):
    """Angles between face forward axis and the face->camera ray (MediaPipe metric frame, camera at origin)."""
    R = T[:3, :3] / np.linalg.norm(T[:3, 0])
    p = T[:3, :3] @ canon[168] + T[:3, 3]             # between the eyes, camera frame
    ray = -p / np.linalg.norm(p)                      # from face towards camera
    # express forward axis in a frame whose z is the ray
    z = ray
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    f = np.array([x, y, z]) @ R[:, 2]
    return math.degrees(math.atan2(f[0], f[2])), math.degrees(math.atan2(f[1], math.hypot(f[0], f[2])))


print("| Face centre x (px) | bearing off optical axis | Tasks matrix yaw, pitch (whole frame) | ORT crop->full-frame: camera-frame yaw, pitch | ORT: yaw, pitch relative to line of sight |")
print("|---|---|---|---|---|")
fw = 300
s = fw / ref_w
small = cv2.resize(IMG, (int(IMG.shape[1] * s), int(IMG.shape[0] * s)), interpolation=cv2.INTER_AREA)
for cx in [960, 1300, 1600, 1800]:
    frame = np.full((H, W, 3), 110, np.uint8)
    x0, y0 = int(cx - small.shape[1] / 2), int(540 - small.shape[0] * 0.35)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + small.shape[1]), min(H, y0 + small.shape[0])
    frame[ya:yb, xa:xb] = small[ya - y0:yb - y0, xa - x0:xb - x0]
    r = tasks.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    tk = "%+.1f, %+.1f" % fl.yaw_pitch(np.array(r.facial_transformation_matrixes[0])) if r.face_landmarks else "not found"
    side = 4 * fw
    c0x, c0y = int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(540 - side / 2, 0, H - side)) if side < H else 0
    side_h = min(side, H)
    crop = frame[c0y:c0y + side_h, c0x:c0x + side]
    d = fl.detect(ds, "short", crop)
    lm = fl.landmarks(lmk, crop, fl.roi_from_detection(d[0]))[0]
    lm = fl.landmarks(lmk, crop, fl.roi_from_landmarks(lm * [crop.shape[1], crop.shape[0], 1]))[0]
    full = np.column_stack([(lm[:, 0] * crop.shape[1] + c0x) / W, (lm[:, 1] * crop.shape[0] + c0y) / H,
                            lm[:, 2] * crop.shape[1] / W])
    T = fl.mp_pose_matrix(full, W, H, canon, weights)
    fpx = (H / 2) / math.tan(math.radians(31.5))
    print(f"| {cx} | {math.degrees(math.atan((cx - 960) / fpx)):+.1f} (eyes at y={full[168,1]*H:.0f}) | {tk} | %+.1f, %+.1f | %+.1f, %+.1f |"
          % (fl.yaw_pitch(T) + los_yaw_pitch(T)))
