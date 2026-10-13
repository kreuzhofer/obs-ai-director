#!/bin/sh
# Convert the MediaPipe Face Landmarker / Face Detector .tflite files to ONNX and
# produce CoreML-EP-friendly variants. Throwaway research script for ticket #15.
#
# usage: convert.sh <models dir>
#   <models dir> must contain face_landmarker.task and blaze_face_full_range.tflite
#   (from storage.googleapis.com/mediapipe-models). Run inside an env with
#   tf2onnx, tensorflow, onnx and onnxruntime installed (e.g. `uv run`).
set -e
D=$1
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$D"
unzip -o -q face_landmarker.task   # face_detector.tflite, face_landmarks_detector.tflite, face_blendshapes.tflite, geometry_pipeline_metadata_landmarks.binarypb
for m in face_detector face_landmarks_detector face_blendshapes blaze_face_full_range; do
  python -m tf2onnx.convert --tflite "$m.tflite" --output "$m.onnx" --opset 17
done
python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param unk__2002 --dim_value 1 \
  face_landmarks_detector.onnx face_landmarks_detector_b1.onnx
python "$HERE/coreml_fix.py" face_landmarks_detector_b1.onnx face_landmarks_detector_cml.onnx
python "$HERE/coreml_fix.py" face_detector.onnx face_detector_cml.onnx
python "$HERE/coreml_fix.py" blaze_face_full_range.onnx blaze_face_full_range_cml.onnx
python "$HERE/coreml_fix.py" face_blendshapes.onnx face_blendshapes_cml.onnx
# canonical face + Procrustes weights (text form of geometry_pipeline_metadata_landmarks.binarypb)
curl -sSL -o meta.pbtxt https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt
