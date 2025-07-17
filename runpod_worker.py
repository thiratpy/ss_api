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

# === 1. Load ALL Models (This runs once when the worker starts) ===

print("🚀 Worker starting, loading models...")
a
# --- MediaPipe & Scikit-learn models for 'predict' endpoint ---
# You'd typically bundle these files with your worker or download them here
MODEL_PATH_MEDIAPIPE = "model/mlp_mediapipe_stroke.pkl"
SCALER_PATH_MEDIAPIPE = "model/scaler_mediapipe.pkl"

mlp_model = joblib.load(MODEL_PATH_MEDIAPIPE)
scaler = joblib.load(SCALER_PATH_MEDIAPIPE)
mp_face_mesh = mp.solutions.face_mesh
# Using min_detection_confidence for better reliability on serverless
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
print("✅ MediaPipe & MLP models loaded.")

# --- PyTorch model for 'predict2' endpoint ---
MODEL_PATH_TORCH = "model/efficientnet_stroke.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = torch.jit.load(MODEL_PATH_TORCH, map_location=device)
torch_model.eval() # Set to evaluation mode, super important for performance
print(f"✅ PyTorch Model loaded on {device}.")

# Preprocessing for the PyTorch model
torch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
torch_classes = ['hemorrhagic', 'ischaemic']

print("✨ All models loaded. Worker is ready! ✨")


# === 2. Helper function for Landmark Extraction ===
def extract_landmarks_from_image(image: Image.Image):
    """Extracts MediaPipe landmarks from a PIL image."""
    # Convert to RGB numpy array, mediapipe needs this format
    np_img = np.array(image.convert("RGB"))
    
    results = face_mesh.process(np_img)
    if not results.multi_face_landmarks:
        return None
    
    # Flatten landmarks into a single list [x1, y1, x2, y2, ...]
    landmark_list = []
    for face_lms in results.multi_face_landmarks:
        for lm in face_lms.landmark:
            landmark_list.extend([lm.x, lm.y])
            
    return landmark_list


# === 3. The RunPod Handler Function ===
def handler(job):
    """
    This is the main function that RunPod calls for each job.
    It checks the 'endpoint' in the input to decide which model to use.
    """
    job_input = job.get('input', {})
    
    # Input validation
    endpoint = job_input.get('endpoint')
    image_b64 = job_input.get('image_base64')

    if not endpoint or not image_b64:
        return {"error": "Missing 'endpoint' or 'image_base64' in input."}
        
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return {"error": f"Failed to decode or open image: {str(e)}"}

    # --- Route to the correct prediction logic based on 'endpoint' ---
    
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
            # The PyTorch model needs an RGB image
            image_rgb = img.convert('RGB')
            
            # Preprocess the image and move it to the correct device
            image_tensor = torch_transform(image_rgb).unsqueeze(0).to(device)

            with torch.no_grad(): # No need to calculate gradients, saves memory and compute
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