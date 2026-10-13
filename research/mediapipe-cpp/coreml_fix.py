"""Make tf2onnx output of the MediaPipe face models CoreML-EP friendly.

1. Give every Conv/MaxPool an explicit `pads` attribute (ORT's CoreML MLProgram builder
   fails with "Required param 'pad' is missing" when the attribute is absent).
2. Rewrite PRelu(x, a) as Relu(x) + a * (x - Relu(x))  (the CoreML EP supports neither PRelu nor Neg).
3. Pin any dynamic batch dim to 1.

usage: coreml_fix.py in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
inits = {i.name: i for i in g.initializer}
for vi in list(g.input) + list(g.output):
    d = vi.type.tensor_type.shape.dim
    if d and not d[0].dim_value:
        d[0].ClearField("dim_param"); d[0].dim_value = 1

new_nodes, n_pads, n_prelu = [], 0, 0
for n in g.node:
    if n.op_type in ("Conv", "MaxPool", "AveragePool") and not any(a.name in ("pads", "auto_pad") for a in n.attribute):
        n.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0])); n_pads += 1
    if n.op_type == "PRelu":
        x, a = n.input
        y = n.output[0]
        p = f"{n.name or y}_prelu"
        # y = Relu(x) + a * (x - Relu(x))   (only Relu/Sub/Mul/Add; the CoreML EP rejects Neg)
        new_nodes += [helper.make_node("Relu", [x], [p + "_pos"]),
                      helper.make_node("Sub", [x, p + "_pos"], [p + "_neg"]),
                      helper.make_node("Mul", [p + "_neg", a], [p + "_scaled"]),
                      helper.make_node("Add", [p + "_pos", p + "_scaled"], [y])]
        n_prelu += 1
        continue
    new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print(f"{sys.argv[1]}: added pads to {n_pads} nodes, rewrote {n_prelu} PRelu")
