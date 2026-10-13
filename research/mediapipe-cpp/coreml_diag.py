"""Does the ORT CoreML EP accept a model, and how much of it?  usage: coreml_diag.py <onnx dir> model.onnx ..."""
import pathlib
import re
import sys

import onnxruntime as ort

D = pathlib.Path(sys.argv[1])
for name in sys.argv[2:]:
    for fmt in ["MLProgram", "NeuralNetwork"]:
        so = ort.SessionOptions()
        so.log_severity_level = 1
        try:
            s = ort.InferenceSession(str(D / name), so, providers=[
                ("CoreMLExecutionProvider", {"ModelFormat": fmt, "RequireStaticInputShapes": "1"}),
                "CPUExecutionProvider"])
            print(f"{name} {fmt}: OK", flush=True)
        except Exception as e:
            print(f"{name} {fmt}: FAIL {re.sub(r'\s+', ' ', str(e))[:160]}", flush=True)
