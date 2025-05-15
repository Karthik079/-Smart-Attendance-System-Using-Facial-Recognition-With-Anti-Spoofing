import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize shared models
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_embedding_from_image(img_rgb):
    """
    Given an RGB image, detect, align, and extract a normalized face embedding.
    """
    face = mtcnn(img_rgb)
    if face is None:
        print("No face detected in the provided image.")
        return None
    # Ensure a batch dimension exists
    if face.dim() == 3:
        face = face.unsqueeze(0)
    with torch.no_grad():
        embedding = facenet(face)
    embedding = embedding.squeeze().numpy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def extract_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return extract_face_embedding_from_image(img_rgb)
