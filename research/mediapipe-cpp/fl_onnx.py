"""Re-implementation of the MediaPipe Face Landmarker graph on ONNX Runtime.

Throwaway research code for ticket #15. Mirrors what a C++ plugin would do:
  BlazeFace (short- or full-range) -> rotated ROI -> FaceMesh-V2 (478 lmk)
  -> head pose either by porting MediaPipe's geometry pipeline (weighted
  Procrustes against the canonical face) or by cv2.solvePnP.

Parameters are taken from the MediaPipe sources (master, fetched 2026-10-13):
  tasks/cc/vision/face_detector/face_detector_graph.cc        (anchors, decode, NMS, 1.5x rect)
  tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc
  tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc     (screen->metric, Procrustes)
  tasks/cc/vision/face_geometry/libs/procrustes_solver.cc
  tasks/cc/vision/face_geometry/face_geometry_from_landmarks_graph.cc (63 deg vFOV, near 1)
"""
import math
import re

import cv2
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------- sessions

def session(path, ep="cpu", units="ALL", fmt="MLProgram", threads=0):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    if ep == "coreml":
        prov = [("CoreMLExecutionProvider", {"ModelFormat": fmt, "MLComputeUnits": units,
                                             "RequireStaticInputShapes": "1"}),
                "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=prov)


# ---------------------------------------------------------------- BlazeFace

def ssd_anchors(size, strides, interpolated):
    """SsdAnchorsCalculator with fixed_anchor_size=true, aspect_ratios=[1.0]."""
    anchors = []
    i = 0
    while i < len(strides):
        n = 0
        j = i
        while j < len(strides) and strides[j] == strides[i]:
            n += 1 + (1 if interpolated else 0)
            j += 1
        fm = math.ceil(size / strides[i])
        for y in range(fm):
            for x in range(fm):
                for _ in range(n):
                    anchors.append(((x + 0.5) / fm, (y + 0.5) / fm))
        i = j
    return np.array(anchors, np.float32)


DETECTORS = {
    # name: (input size, anchors)
    "short": (128, ssd_anchors(128, [8, 16, 16, 16], True)),   # 896 anchors
    "full": (192, ssd_anchors(192, [4], False)),                # 2304 anchors
}


def letterbox(img, size):
    """keep_aspect_ratio=true, BORDER_ZERO, centred. Returns tensor in [-1,1] and (L, offx, offy)."""
    h, w = img.shape[:2]
    L = max(w, h)
    canvas = np.zeros((L, L, 3), np.uint8)
    ox, oy = (L - w) // 2, (L - h) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    t = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return (t / 127.5 - 1.0)[None], (L, ox, oy)


def detect(sess, kind, img, min_score=0.5, nms_iou=0.3):
    """Returns list of dicts: score, box (cx,cy,w,h px), kps (6x2 px)."""
    size, anchors = DETECTORS[kind]
    x, (L, ox, oy) = letterbox(img, size)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    reg, cls = (outs[0], outs[1]) if outs[0].shape[-1] == 16 else (outs[1], outs[0])
    reg, cls = reg[0], cls[0, :, 0]
    score = 1 / (1 + np.exp(-np.clip(cls, -100, 100)))
    keep = np.where(score >= min_score)[0]
    dets = []
    for k in keep:
        r, (ax, ay) = reg[k] / size, anchors[k]
        cx, cy, bw, bh = r[0] + ax, r[1] + ay, r[2], r[3]
        kps = np.stack([r[4::2][:6] + ax, r[5::2][:6] + ay], 1)
        dets.append((score[k], np.array([cx, cy, bw, bh]), kps))
    dets.sort(key=lambda d: -d[0])
    # weighted NMS (IoU)
    out = []
    while dets:
        s0, b0, _ = dets[0]
        group, rest = [], []
        for d in dets:
            (group if iou(b0, d[1]) > nms_iou else rest).append(d)
        ws = np.array([g[0] for g in group])
        box = sum(w * g[1] for w, g in zip(ws, group)) / ws.sum()
        kp = sum(w * g[2] for w, g in zip(ws, group)) / ws.sum()
        out.append({"score": float(s0),
                    "box": np.array([box[0] * L - ox, box[1] * L - oy, box[2] * L, box[3] * L]),
                    "kps": np.stack([kp[:, 0] * L - ox, kp[:, 1] * L - oy], 1)})
        dets = rest
    return out


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def roi_from_detection(det):
    """DetectionsToRects (rotation from kp0->kp1, target 0) + RectTransformation scale 1.5."""
    (x0, y0), (x1, y1) = det["kps"][0], det["kps"][1]
    angle = -math.atan2(-(y1 - y0), x1 - x0)
    cx, cy, w, h = det["box"]
    return (cx, cy, w * 1.5, h * 1.5, angle)


def roi_from_landmarks(lm_px):
    """Tracking ROI for the next frame (eye corners 33/263, bbox, scale 1.5, square_long)."""
    (x0, y0), (x1, y1) = lm_px[33, :2], lm_px[263, :2]
    angle = -math.atan2(-(y1 - y0), x1 - x0)
    c, s = math.cos(-angle), math.sin(-angle)
    p = lm_px[:, :2]
    mid = (p.min(0) + p.max(0)) / 2
    q = (p - mid) @ np.array([[c, s], [-s, c]])
    lo, hi = q.min(0), q.max(0)
    cq = (lo + hi) / 2
    centre = mid + cq @ np.array([[c, -s], [s, c]])
    side = max(hi - lo) * 1.5
    return (centre[0], centre[1], side, side, angle)


# ---------------------------------------------------------------- FaceMesh-V2

def crop(img, roi, size=256):
    cx, cy, w, h, a = roi
    c, s = math.cos(a), math.sin(a)
    # maps output pixel (u,v) -> image pixel
    M = np.array([[c * w / size, -s * h / size, 0], [s * w / size, c * h / size, 0]], np.float64)
    M[:, 2] = np.array([cx, cy]) - M[:, :2] @ np.array([size / 2, size / 2])
    patch = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT)
    return patch, M


def landmarks(sess, img, roi, size=256):
    """Returns (478,3) landmarks as MediaPipe NormalizedLandmarks (x,y in [0,1] of image, z ~ x scale)
    plus the face-presence score."""
    h_img, w_img = img.shape[:2]
    patch, M = crop(img, roi, size)
    x = (patch.astype(np.float32) / 255.0)[None]
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    raw = next(o for o in outs if o.size == 1434).reshape(478, 3)
    pres = next(o for o in outs if o.size == 1 and o.ndim == 4)
    xy = raw[:, :2] @ M[:, :2].T + M[:, 2]
    z = raw[:, 2] * (roi[2] / size) / w_img
    out = np.column_stack([xy[:, 0] / w_img, xy[:, 1] / h_img, z])
    return out, float(1 / (1 + np.exp(-pres.ravel()[0])))


BLEND_IDX = [0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66,
             67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145,
             146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176,
             178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285,
             288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
             336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387,
             388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474,
             475, 476, 477]


def blend_input(lm_norm, w, h):
    return (lm_norm[BLEND_IDX, :2] * np.array([w, h], np.float32)).astype(np.float32)[None]


# ---------------------------------------------------------------- geometry pipeline port

def load_geometry(pbtxt_path):
    txt = open(pbtxt_path).read()
    basis = [(int(a), float(b)) for a, b in re.findall(r"landmark_id: (\d+) weight: ([0-9.eE-]+)", txt)]
    vb = np.array([float(v) for v in re.findall(r"vertex_buffer: ([-0-9.eE]+)", txt)], np.float64)
    canon = vb.reshape(-1, 5)[:, :3]           # VERTEX_PT: x y z u v
    weights = np.zeros(len(canon))
    for i, wgt in basis:
        weights[i] = wgt
    return canon, weights


def procrustes(src, dst, w):
    """FloatPrecisionProcrustesSolver::SolveWeightedOrthogonalProblem (src, dst: 3xN)."""
    sw = np.sqrt(w)
    ws, wt = src * sw, dst * sw
    tw = (sw * sw).sum()
    com = (ws * sw).sum(1) / tw
    cws = ws - np.outer(com, sw)
    U, _, Vt = np.linalg.svd(wt @ cws.T)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, 2] *= -1
    R = U @ Vt
    scale = (R @ cws * wt).sum() / (cws * ws).sum()
    RS = scale * R
    t = ((wt - RS @ ws) * sw).sum(1) / tw
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = RS, t
    return T


def mp_pose_matrix(lm_norm, w, h, canon, weights, vfov=63.0, near=1.0):
    """GeometryPipeline: screen landmarks (468) -> 4x4 pose transform (matches Tasks output)."""
    hn = 2 * near * math.tan(math.radians(vfov) / 2)
    wn = w * hn / h
    left, bottom = -wn / 2, -hn / 2
    s = lm_norm[:468].T.astype(np.float64).copy()
    s[1] = 1 - s[1]                              # TOP_LEFT_CORNER origin
    s = s * np.array([[wn], [hn], [wn]]) + np.array([[left], [bottom], [0]])
    depth = s[2].mean()
    src = canon.T

    def est_scale(m):
        return np.linalg.norm(procrustes(src, m, weights)[:3, 0])

    def flip(m):
        m = m.copy(); m[2] *= -1; return m

    def move_unproject(m, sc):
        m = m.copy()
        m[2] = (m[2] - depth + near) / sc
        m[0] = m[0] * m[2] / near
        m[1] = m[1] * m[2] / near
        return m

    s1 = est_scale(flip(s))
    s2 = est_scale(flip(move_unproject(s, s1)))
    metric = flip(move_unproject(s, s1 * s2))
    return procrustes(src, metric, weights)


# ---------------------------------------------------------------- solvePnP alternative

F = np.diag([1.0, -1.0, -1.0])   # OpenCV camera (y down, z fwd) <-> MediaPipe metric (y up, z to viewer)


def pnp_pose(lm_norm, w, h, canon, vfov=63.0, idx=None, method=cv2.SOLVEPNP_SQPNP, refine=True):
    """Rotation of the canonical face in MediaPipe's metric camera frame, via cv2.solvePnP."""
    idx = np.arange(468) if idx is None else np.asarray(idx)
    f = (h / 2) / math.tan(math.radians(vfov) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    obj = (canon[idx] @ F).astype(np.float64)                   # canonical expressed in OpenCV object axes
    img = (lm_norm[idx, :2] * np.array([w, h])).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=method)
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    Rcv, _ = cv2.Rodrigues(rvec)
    return F @ Rcv @ F, tvec.ravel()


def yaw_pitch(R):
    """Same convention as prototype/attention_detector.py (forward axis of the rotation)."""
    f = np.asarray(R)[:3, 2]
    return math.degrees(math.atan2(f[0], f[2])), math.degrees(math.atan2(f[1], math.hypot(f[0], f[2])))
