import cv2
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from deepface import DeepFace  
from datetime import datetime, timedelta
from face_detection import detect_faces  
from face_embedding import extract_face_embedding_from_image  
from attendance import record_attendance
from embedding_retrieval import EmbeddingRetrieval
from collections import deque
import torch

# Track last attendance
last_attendance = {}

# Directories
SPOOF_DIR = "data/spoof_attempts"
os.makedirs(SPOOF_DIR, exist_ok=True)

class FaceRecognitionBuffer:
    def __init__(self, buffer_size=10, min_confidence=0.6):  # Reduced buffer size
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.predictions = deque(maxlen=buffer_size)
        self.current_identity = "Unknown"
        self.confidence_score = 0.0

    def update(self, name, distance):
        # Convert distance to confidence score (inverse relationship)
        confidence = 1.0 / (1.0 + distance)
        
        # Only consider predictions with high confidence
        if confidence > self.min_confidence:
            self.predictions.append(name)
            
            # Get the most common prediction in the buffer
            if len(self.predictions) >= 3:  # Reduced minimum predictions needed
                prediction_counts = {}
                for pred in self.predictions:
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                
                # Find the most common prediction
                max_count = 0
                max_pred = "Unknown"
                for pred, count in prediction_counts.items():
                    if count > max_count:
                        max_count = count
                        max_pred = pred
                
                # Calculate confidence as ratio of predictions
                confidence_ratio = max_count / len(self.predictions)
                
                # Update only if we have a stable prediction
                if confidence_ratio > 0.6:  # Reduced threshold for faster recognition
                    self.current_identity = max_pred
                    self.confidence_score = confidence_ratio
                    return True
        
        return False

    def get_current_prediction(self):
        return self.current_identity, self.confidence_score

class OptimizedRecognition:
    def __init__(self):
        # Initialize models
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                          thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = self.facenet.to(self.device)
        
        # Pre-normalize known embeddings
        self.known_embeddings = {}
        self.normalized_embeddings = {}
        self.embedding_retrieval = EmbeddingRetrieval()
        self.last_embeddings_update = None
        self.update_interval = timedelta(minutes=5)
        
        # Initialize face buffer
        self.face_buffer = FaceRecognitionBuffer()
        
    def update_known_embeddings(self, force=False):
        current_time = datetime.now()
        if (not force and self.last_embeddings_update is not None and 
            current_time - self.last_embeddings_update < self.update_interval):
            return
            
        self.known_embeddings = self.embedding_retrieval.get_all_embeddings()
        
        # Pre-normalize all embeddings
        self.normalized_embeddings = {}
        for user_id, embeddings in self.known_embeddings.items():
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.normalized_embeddings[user_id] = embeddings / (norms + 1e-10)
        
        self.last_embeddings_update = current_time

    def process_frame(self, frame, threshold=0.7):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        try:
            boxes, _ = self.mtcnn.detect(rgb_frame)
            if boxes is None:
                return frame
                
            # Update embeddings periodically
            self.update_known_embeddings()
            
            for box in boxes:
                x, y, w, h = map(int, [box[0], box[1], box[2] - box[0], box[3] - box[1]])
                
                # Extract and normalize face embedding
                face_crop = frame[y:y+h, x:x+w]
                embedding = extract_face_embedding_from_image(face_crop)
                
                if embedding is not None:
                    # Normalize current embedding
                    embedding_norm = np.linalg.norm(embedding)
                    if embedding_norm > 0:
                        embedding = embedding / embedding_norm
                    
                    # Find best match
                    min_dist = float('inf')
                    best_match = "Unknown"
                    
                    # Vectorized distance calculation
                    for user_id, norm_embeddings in self.normalized_embeddings.items():
                        similarities = np.dot(norm_embeddings, embedding)
                        max_similarity = np.max(similarities)
                        person_min_dist = 1 - max_similarity
                        
                        if person_min_dist < min_dist:
                            min_dist = person_min_dist
                            best_match = user_id
                    
                    # Update recognition buffer
                    if self.face_buffer.update(best_match, min_dist):
                        recognized_name, confidence_score = self.face_buffer.get_current_prediction()
                    else:
                        recognized_name = "Unknown"
                        confidence_score = 0.0
                    
                    # Record attendance if confident
                    if recognized_name != "Unknown" and confidence_score > threshold:
                        current_time = datetime.now()
                        if (recognized_name not in last_attendance or
                            current_time - last_attendance[recognized_name] > timedelta(minutes=40)):
                            record_attendance(recognized_name)
                            last_attendance[recognized_name] = current_time
                            self.update_known_embeddings(force=True)
                    
                    # Draw results
                    label = f"{recognized_name} ({confidence_score:.2f})" if recognized_name != "Unknown" else "Unknown"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return frame

def recognize_face_live(threshold=0.7, cooldown_minutes=40):
    recognition = OptimizedRecognition()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting live face recognition. Press 'q' to quit.")
    print("Loading embeddings from MongoDB...")
    
    # Initial embeddings load
    recognition.update_known_embeddings(force=True)
    
    print(f"Loaded embeddings for {len(recognition.known_embeddings)} users")
    print(f"Using device: {recognition.device}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        # Process frame
        processed_frame = recognition.process_frame(frame, threshold)

        # Display frame
        cv2.imshow("Live Recognition", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    recognize_face_live()
