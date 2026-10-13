# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mediapipe",
#   "numpy",
#   "psutil",
#   "onnxruntime",
#   "pyobjc-framework-Vision; sys_platform == 'darwin'",
#   "pyobjc-framework-Quartz; sys_platform == 'darwin'",
# ]
# ///
"""PROTOTYPE, throwaway. Wayfinder ticket #6 "Prototype: Standalone Attention detector".

Answers: can one webcam tell whether the Presenter is attending to it, and can
it tell "into the lens" from "at a nearby screen"? Not production code.

Run:   uv run prototype/attention_detector.py [--camera 0]
Test:  uv run prototype/attention_detector.py --selftest

Keys (in the video window):
  z        zero: treat the current pose/gaze as "looking into the lens"
  d        cycle analysis downscale 1 -> 1/2 -> 1/4 -> 1/8 (simulates a distant Presenter / small render target)
  c        toggle full-resolution face-crop pass (find face in the small frame, re-run on a full-res crop)
  g        toggle L2CS gaze model (yakhyo MobileOne-S0 ONNX)
  p        switch L2CS ONNX Runtime provider: CPU <-> CoreML
  v        toggle Apple Vision head pose (macOS only)
  1 2 3    start recording labelled samples: 1 = into lens, 2 = at a nearby screen, 3 = away
  0        stop recording
  q / Esc  quit and print the separability summary
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

HERE = Path(__file__).parent
MODELS = HERE / ".models"
RUNS = HERE / "runs"
FACE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
GAZE_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx"
SAMPLE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"

LABELS = {ord("1"): "lens", ord("2"): "screen", ord("3"): "away"}
SCALES = [1.0, 0.5, 0.25, 0.125]
SIGNALS = ["head_yaw", "head_pitch", "iris_h", "iris_v", "blend_h", "blend_v",
           "l2cs_yaw", "l2cs_pitch", "vis_yaw", "vis_pitch", "face_px"]


def fetch(url, name):
    MODELS.mkdir(exist_ok=True)
    path = MODELS / name
    if not path.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


# ---------------------------------------------------------------- MediaPipe

def make_landmarker(mode):
    opts = mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(fetch(FACE_TASK_URL, "face_landmarker.task"))),
        running_mode=mode,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mpv.FaceLandmarker.create_from_options(opts)


def mp_signals(result, w, h):
    """Head yaw/pitch (deg), iris offsets, blendshape eye offsets, face bbox (px), landmarks (px)."""
    if not result.face_landmarks:
        return None
    lm = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
    R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
    f = R[:, 2]  # canonical face forward axis (+z points at the camera)
    out = {
        "head_yaw": math.degrees(math.atan2(f[0], f[2])),
        "head_pitch": math.degrees(math.atan2(f[1], math.hypot(f[0], f[2]))),
    }

    def eye(iris, a, b, top, bot):
        x0, x1 = sorted((lm[a, 0], lm[b, 0]))
        hr = (lm[iris, 0] - x0) / max(x1 - x0, 1e-6) - 0.5
        vr = (lm[iris, 1] - lm[top, 1]) / max(lm[bot, 1] - lm[top, 1], 1e-6) - 0.5
        return hr, vr

    rh, rv = eye(468, 33, 133, 159, 145)
    lh, lv = eye(473, 362, 263, 386, 374)
    out["iris_h"], out["iris_v"] = (rh + lh) / 2, (rv + lv) / 2

    b = {c.category_name: c.score for c in result.face_blendshapes[0]}
    out["blend_h"] = (b["eyeLookOutLeft"] + b["eyeLookInRight"] - b["eyeLookInLeft"] - b["eyeLookOutRight"]) / 2
    out["blend_v"] = (b["eyeLookUpLeft"] + b["eyeLookUpRight"] - b["eyeLookDownLeft"] - b["eyeLookDownRight"]) / 2
    out["face_px"] = float(np.linalg.norm(lm[468] - lm[473]))  # inter-pupil distance in analysed pixels

    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    return out, (x0, y0, x1, y1), lm


def crop_box(box, w, h, margin):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + margin) / 2
    return (int(max(cx - s, 0)), int(max(cy - s, 0)), int(min(cx + s, w)), int(min(cy + s, h)))


# ---------------------------------------------------------------- L2CS gaze (ONNX)

class Gaze:
    def __init__(self):
        import onnxruntime as ort
        self.ort = ort
        self.path = str(fetch(GAZE_URL, "mobileone_s0_gaze.onnx"))
        self.provider = "CPUExecutionProvider"
        self._load()

    def _load(self):
        providers = [self.provider] + (["CPUExecutionProvider"] if self.provider != "CPUExecutionProvider" else [])
        self.sess = self.ort.InferenceSession(self.path, providers=providers)

    def toggle_provider(self):
        avail = self.ort.get_available_providers()
        self.provider = "CoreMLExecutionProvider" if self.provider == "CPUExecutionProvider" and "CoreMLExecutionProvider" in avail else "CPUExecutionProvider"
        self._load()

    def __call__(self, face_bgr):
        img = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), (448, 448)).astype(np.float32) / 255
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = img.transpose(2, 0, 1)[None].astype(np.float32)
        yaw_l, pitch_l = self.sess.run(None, {"input": x})
        idx = np.arange(90, dtype=np.float32)

        def dec(l):
            e = np.exp(l - l.max(1, keepdims=True))
            return float((e / e.sum(1, keepdims=True) * idx).sum() * 4 - 180)

        return {"l2cs_yaw": dec(yaw_l), "l2cs_pitch": dec(pitch_l)}


# ---------------------------------------------------------------- Apple Vision

class AppleVision:
    def __init__(self):
        import Quartz
        import Vision
        self.Q, self.V = Quartz, Vision

    def __call__(self, bgr):
        Q, V = self.Q, self.V
        h, w = bgr.shape[:2]
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).tobytes()
        provider = Q.CGDataProviderCreateWithData(None, rgba, len(rgba), None)
        cg = Q.CGImageCreate(w, h, 8, 32, w * 4, Q.CGColorSpaceCreateDeviceRGB(),
                             Q.kCGImageAlphaNoneSkipLast, provider, None, False, Q.kCGRenderingIntentDefault)
        req = V.VNDetectFaceRectanglesRequest.alloc().init()
        req.setRevision_(3)
        handler = V.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})
        handler.performRequests_error_([req], None)
        res = req.results()
        if not res:
            return None
        face = max(res, key=lambda o: o.boundingBox().size.width)
        out = {"vis_conf": float(face.confidence())}
        if face.yaw() is not None:
            out["vis_yaw"] = math.degrees(float(face.yaw()))
        if face.pitch() is not None:
            out["vis_pitch"] = math.degrees(float(face.pitch()))
        return out


# ---------------------------------------------------------------- analysis of one frame

class Detector:
    def __init__(self, use_gaze=True, use_vision=True):
        self.video = make_landmarker(mpv.RunningMode.VIDEO)
        self.image = make_landmarker(mpv.RunningMode.IMAGE)
        self.gaze = Gaze() if use_gaze else None
        self.vision = AppleVision() if use_vision and sys.platform == "darwin" else None
        self.gaze_on, self.vision_on = bool(self.gaze), bool(self.vision)
        self.scale_i, self.crop = 0, False
        self.ts = 0
        self.last_box = None
        self.last_box_age = 99

    def analyse(self, frame):
        H, W = frame.shape[:2]
        s = SCALES[self.scale_i]
        small = cv2.resize(frame, (max(int(W * s), 1), max(int(H * s), 1)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        sh, sw = small.shape[:2]
        ms, sig, draw = {}, {}, {}

        t = time.perf_counter()
        self.ts += 33
        r = self.video.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), self.ts)
        ms["mp"] = (time.perf_counter() - t) * 1000
        found = mp_signals(r, sw, sh)

        analysed, ax0, ay0, k = small, 0, 0, 1 / s  # analysed image, its offset and scale in full-res px
        if found:
            self.last_box, self.last_box_age = tuple(v / s for v in found[1]), 0
        else:
            self.last_box_age += 1

        if self.crop and self.last_box and self.last_box_age < 15:
            x0, y0, x1, y1 = crop_box(self.last_box, W, H, 0.6)
            crop = frame[y0:y1, x0:x1]
            t = time.perf_counter()
            r = self.image.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
            ms["mp_crop"] = (time.perf_counter() - t) * 1000
            found = mp_signals(r, crop.shape[1], crop.shape[0])
            analysed, ax0, ay0, k = crop, x0, y0, 1
            draw["crop"] = (x0, y0, x1, y1)

        if found:
            sig.update(found[0])
            bx0, by0, bx1, by1 = found[1]
            draw["box"] = (ax0 + bx0 * k, ay0 + by0 * k, ax0 + bx1 * k, ay0 + by1 * k)
            draw["iris"] = [(ax0 + found[2][i, 0] * k, ay0 + found[2][i, 1] * k) for i in (468, 473)]
            if self.gaze_on:
                ah, aw = analysed.shape[:2]
                cx0, cy0, cx1, cy1 = crop_box(found[1], aw, ah, 0.25)
                face = analysed[cy0:cy1, cx0:cx1]
                if face.size:
                    t = time.perf_counter()
                    sig.update(self.gaze(face))
                    ms["l2cs"] = (time.perf_counter() - t) * 1000

        if self.vision_on:
            t = time.perf_counter()
            v = self.vision(analysed)
            ms["vision"] = (time.perf_counter() - t) * 1000
            if v:
                sig.update(v)
        return sig, ms, draw, (sw, sh)


# ---------------------------------------------------------------- verdicts and stats

def verdicts(c, a):
    """c: calibrated signals. Each verdict is True/False/None (no data)."""
    def within(keys, th):
        if any(k not in c for k in keys):
            return None
        return all(abs(c[k]) < th for k in keys)

    head = within(["head_yaw", "head_pitch"], a.head_deg)
    iris = within(["iris_h", "iris_v"], a.iris)
    blend = within(["blend_h", "blend_v"], a.blend)
    return {
        "MP head only": head,
        "MP head + iris": None if head is None or iris is None else head and iris,
        "MP head + blendshapes": None if head is None or blend is None else head and blend,
        "L2CS gaze": within(["l2cs_yaw", "l2cs_pitch"], a.gaze_deg),
        "Vision head": within(["vis_yaw", "vis_pitch"], a.head_deg),
    }


def summary(samples):
    lines = []
    for sname in SIGNALS:
        row, stats = [], {}
        for lab in ("lens", "screen", "away"):
            vals = [x[sname] for x in samples.get(lab, []) if sname in x]
            if vals:
                stats[lab] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                row.append(f"{lab} {stats[lab][0]:+7.2f}±{stats[lab][1]:5.2f}")
        if not row:
            continue
        sep = ""
        if "lens" in stats and "screen" in stats:
            (m1, s1, _), (m2, s2, _) = stats["lens"], stats["screen"]
            sep = f"   d'(lens/screen)={abs(m1 - m2) / max(math.sqrt((s1 ** 2 + s2 ** 2) / 2), 1e-6):5.2f}"
        lines.append(f"{sname:11s} " + "  ".join(row) + sep)
    return lines


def put(img, text, y, color=(230, 230, 230), x=10, scale=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------- main loops

def selftest(a):
    img = cv2.imread(str(fetch(SAMPLE_URL, "portrait.jpg")))
    det = Detector(not a.no_gaze, not a.no_vision)
    for crop in (False, True):
        det.crop = crop
        for i in range(len(SCALES)):
            det.scale_i = i
            for _ in range(5):
                sig, ms, _, (sw, sh) = det.analyse(img)
            print(f"crop={crop!s:5} scale={SCALES[i]:<6} {sw}x{sh}  " +
                  " ".join(f"{k}={v:+.2f}" for k, v in sig.items()) + "  | " +
                  " ".join(f"{k}={v:.1f}ms" for k, v in ms.items()))
    if det.gaze:
        for prov in ("CPU", "CoreML"):
            det.crop, det.scale_i = False, 0
            for _ in range(20):
                _, ms, _, _ = det.analyse(img)
            print(f"L2CS {det.gaze.provider}: {ms.get('l2cs', float('nan')):.1f} ms")
            det.gaze.toggle_provider()


def live(a):
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {a.camera} (grant the terminal camera access in System Settings > Privacy)")
    det = Detector(not a.no_gaze, not a.no_vision)
    proc = psutil.Process()
    proc.cpu_percent()
    offset = defaultdict(float)
    samples = defaultdict(list)
    recording = None
    frame_times = deque(maxlen=30)
    ms_avg = defaultdict(lambda: deque(maxlen=30))
    cpu = 0.0
    last_cpu = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sig, ms, draw, (sw, sh) = det.analyse(frame)
        frame_times.append(time.perf_counter())
        for k, v in ms.items():
            ms_avg[k].append(v)
        if time.time() - last_cpu > 1:
            cpu, last_cpu = proc.cpu_percent(), time.time()
        cal = {k: v - offset[k] for k, v in sig.items() if k in SIGNALS and k != "face_px"}
        if recording and sig:
            samples[recording].append(dict(sig))
        vd = verdicts(cal, a)

        H, W = frame.shape[:2]
        view = frame.copy()
        if "crop" in draw:
            cv2.rectangle(view, draw["crop"][:2], draw["crop"][2:], (255, 160, 0), 2)
        if "box" in draw:
            x0, y0, x1, y1 = map(int, draw["box"])
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0) if vd.get("MP head + iris") else (0, 0, 255), 2)
            for p in draw["iris"]:
                cv2.circle(view, tuple(map(int, p)), 3, (0, 255, 255), -1)
        dw = 960
        view = cv2.resize(view, (dw, int(H * dw / W)))
        panel = np.zeros((max(view.shape[0], 620), 470, 3), np.uint8)
        if view.shape[0] < panel.shape[0]:
            view = cv2.copyMakeBorder(view, 0, panel.shape[0] - view.shape[0], 0, 0, cv2.BORDER_CONSTANT)

        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
        y = 22
        put(panel, f"fps {fps:5.1f}   process CPU {cpu:5.0f}%  (100% = 1 core)", y); y += 20
        put(panel, f"analysis {sw}x{sh} (scale {SCALES[det.scale_i]})  crop pass {'ON' if det.crop else 'off'}", y); y += 20
        put(panel, "  ".join(f"{k} {np.mean(v):.1f}ms" for k, v in ms_avg.items() if v), y); y += 20
        put(panel, f"L2CS {'ON ' + det.gaze.provider.replace('ExecutionProvider', '') if det.gaze_on else 'off'}"
                   f"   Vision {'ON' if det.vision_on else 'off'}", y); y += 28
        if not sig:
            put(panel, "NO FACE", y, (0, 0, 255), scale=0.8); y += 30
        for k in SIGNALS:
            if k in sig:
                v = sig[k] if k == "face_px" else cal[k]
                unit = "px" if k == "face_px" else ("deg" if "yaw" in k or "pitch" in k else "")
                put(panel, f"{k:11s} {v:+8.2f} {unit}", y); y += 18
        if "vis_conf" in sig:
            put(panel, f"vis_conf    {sig['vis_conf']:8.2f}", y); y += 18
        y += 10
        put(panel, "ATTENDING?", y, scale=0.6); y += 22
        for name, v in vd.items():
            txt, col = ("--", (120, 120, 120)) if v is None else (("YES", (0, 220, 0)) if v else ("no", (0, 0, 255)))
            put(panel, f"{name:22s} {txt}", y, col); y += 20
        y += 10
        rec = f"REC '{recording}'" if recording else "not recording"
        put(panel, f"{rec}   " + " ".join(f"{k}:{len(v)}" for k, v in samples.items()), y, (0, 200, 255) if recording else (200, 200, 200)); y += 20
        for line in summary(samples)[:8]:
            put(panel, line[:70], y, scale=0.38); y += 15
        put(panel, "z zero  d scale  c crop  g L2CS  p provider  v Vision", panel.shape[0] - 30, (150, 150, 150), scale=0.42)
        put(panel, "1 lens  2 screen  3 away  0 stop rec  q quit", panel.shape[0] - 12, (150, 150, 150), scale=0.42)
        cv2.imshow("PROTOTYPE Attention detector", np.hstack([view, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("z"):
            offset = defaultdict(float, {k: v for k, v in sig.items() if k in SIGNALS})
        elif key == ord("d"):
            det.scale_i = (det.scale_i + 1) % len(SCALES)
        elif key == ord("c"):
            det.crop = not det.crop
        elif key == ord("g") and det.gaze:
            det.gaze_on = not det.gaze_on
        elif key == ord("p") and det.gaze:
            det.gaze.toggle_provider()
            ms_avg.pop("l2cs", None)
        elif key == ord("v") and det.vision:
            det.vision_on = not det.vision_on
        elif key in LABELS:
            recording = LABELS[key]
        elif key == ord("0"):
            recording = None

    cap.release()
    cv2.destroyAllWindows()
    if samples:
        print("\n".join(summary(samples)))
        RUNS.mkdir(exist_ok=True)
        out = RUNS / f"samples-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.write_text(json.dumps({"args": vars(a), "offset": offset, "samples": samples}, indent=1))
        print(f"samples written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--head-deg", type=float, default=15, help="head yaw/pitch threshold for 'attending'")
    ap.add_argument("--gaze-deg", type=float, default=10, help="L2CS gaze yaw/pitch threshold")
    ap.add_argument("--iris", type=float, default=0.08, help="iris offset threshold (fraction of eye width/height)")
    ap.add_argument("--blend", type=float, default=0.3, help="eyeLook blendshape offset threshold")
    ap.add_argument("--no-gaze", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run every engine on a sample still image and print values/timings")
    a = ap.parse_args()
    selftest(a) if a.selftest else live(a)
