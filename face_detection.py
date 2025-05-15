
import cv2
import mediapipe as mp
from facenet_pytorch import MTCNN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
mp_face_detection = mp.solutions.face_detection
blazeface = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_faces(image, blaze_confidence_threshold=0.90, mtcnn_confidence_threshold=0.90):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    detected_faces = []

    blazeface_results = blazeface.process(img_rgb)
    if blazeface_results.detections:
        for detection in blazeface_results.detections:
            bbox = detection.location_data.relative_bounding_box
            confidence = detection.score[0]
            
            if confidence > blaze_confidence_threshold:
                x_min = int(bbox.xmin * width)
                y_min = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                detected_faces.append([x_min, y_min, x_min + w, y_min + h])
    
    if not detected_faces or (blazeface_results.detections and confidence < blaze_confidence_threshold):
        mtcnn_results, _ = mtcnn.detect(image)
        if mtcnn_results is not None:
            for box in mtcnn_results:
                x1, y1, x2, y2 = map(int, box)
                detected_faces.append([x1, y1, x2, y2])
    
    unique_faces = []
    for face in detected_faces:
        if face not in unique_faces:
            unique_faces.append(face)
    
    return unique_faces