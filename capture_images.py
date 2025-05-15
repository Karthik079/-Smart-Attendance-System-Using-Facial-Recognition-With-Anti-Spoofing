
import cv2
import os
import time

def capture_training_images(name, save_dir="data/training_images", num_images=100, delay=0.25):
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print(f"Capturing {num_images} images for {name}. Move your face slightly between captures.")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Try again.")
            continue
        
        cv2.imshow("Capture", frame)
        img_path = os.path.join(person_dir, f"{count+1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1
        
        time.sleep(delay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Image capture for {name} completed.")