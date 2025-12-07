import onnx

# Load the original model
model = onnx.load("hair_classifier_v1.onnx")

# Force the IR version to 8 (compatible with onnxruntime 1.14+)
model.ir_version = 8

# Save as a new file
onnx.save(model, "hair_classifier_patched.onnx")
print("Model patched and saved as 'hair_classifier_patched.onnx'")