import os
import cv2
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import mediapipe as mp
import torch
from torchvision import transforms
import runpod
import base64

# === 1. Load ALL Models (with detailed error logging) ===
print("🚀 Worker starting, attempting to load models...")

try:
    # --- MediaPipe & Scikit-learn models for 'predict' endpoint ---
    MODEL_PATH_MEDIAPIPE = "model/mlp_mediapipe_stroke.pkl"
    SCALER_PATH_MEDIAPIPE = "model/scaler_mediapipe.pkl"

    # Check if files exist before loading
    if not os.path.exists(MODEL_PATH_MEDIAPIPE):
        raise FileNotFoundError(f"FATAL: MediaPipe model not found at {MODEL_PATH_MEDIAPIPE}")
    if not os.path.exists(SCALER_PATH_MEDIAPIPE):
        raise FileNotFoundError(f"FATAL: MediaPipe scaler not found at {SCALER_PATH_MEDIAPIPE}")

    mlp_model = joblib.load(MODEL_PATH_MEDIAPIPE)
    scaler = joblib.load(SCALER_PATH_MEDIAPIPE)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    print("✅ MediaPipe & MLP models loaded successfully.")

    # --- PyTorch model for 'predict2' endpoint ---
    MODEL_PATH_TORCH = "model/efficientnet_stroke.pt"

    if not os.path.exists(MODEL_PATH_TORCH):
        raise FileNotFoundError(f"FATAL: PyTorch model not found at {MODEL_PATH_TORCH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch.jit.load(MODEL_PATH_TORCH, map_location=device)
    torch_model.eval()
    print(f"✅ PyTorch Model loaded successfully on {device}.")

    # Preprocessing for the PyTorch model
    torch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    torch_classes = ['hemorrhagic', 'ischaemic']

    print("✨ All models loaded. Worker is ready! ✨")

except Exception as e:
    # This will print the exact error and stop the worker gracefully if loading fails
    print(f"❌❌❌ MODEL LOADING FAILED ❌❌❌")
    print(f"Error: {e}")
    # Exit so RunPod knows the worker failed to initialize
    exit(1)


# === 2. Helper function for Landmark Extraction ===
# (This part is unchanged)
def extract_landmarks_from_image(image: Image.Image):
    np_img = np.array(image.convert("RGB"))
    results = face_mesh.process(np_img)
    if not results.multi_face_landmarks:
        return None
    landmark_list = []
    for face_lms in results.multi_face_landmarks:
        for lm in face_lms.landmark:
            landmark_list.extend([lm.x, lm.y])
    return landmark_list


# === 3. The RunPod Handler Function ===
# (This part is unchanged)
def handler(job):
    job_input = job.get('input', {})
    endpoint = job_input.get('endpoint')
    image_b64 = job_input.get('image_base64')

    if not endpoint or not image_b64:
        return {"error": "Missing 'endpoint' or 'image_base64' in input."}
        
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to decode or open image: {str(e)}"}

    if endpoint == 'predict':
        try:
            landmark_vector = extract_landmarks_from_image(img)
            if not landmark_vector:
                return {"error": "No face detected or landmarks could not be extracted."}
            X = np.array(landmark_vector).reshape(1, -1)
            X_scaled = scaler.transform(X)
            stroke_prob = mlp_model.predict_proba(X_scaled)[0][1]
            prediction = mlp_model.predict(X_scaled)[0]
            label = "stroke" if prediction == 1 else "noStroke"
            return {
                "label": label,
                "stroke_probability": round(float(stroke_prob), 4),
                "message": "✅ Prediction successful (MediaPipe)"
            }
        except Exception as e:
            return {"error": f"An error occurred during landmark prediction: {str(e)}"}

    elif endpoint == 'predict2':
        try:
            image_rgb = img.convert('RGB')
            image_tensor = torch_transform(image_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = torch_model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_index = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_index].item()
            return {
                'prediction': torch_classes[pred_index],
                'confidence': f"{confidence:.4f}",
                "message": "✅ Prediction successful (PyTorch)"
            }
        except Exception as e:
            return {"error": f"An error occurred during image classification: {str(e)}"}
    
    else:
        return {"error": f"Invalid endpoint '{endpoint}'. Use 'predict' or 'predict2'."}


# === 4. Start the serverless worker ===
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})