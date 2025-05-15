from capture_images import capture_training_images
from train_model import train_faces
from recognition import recognize_face_live

if __name__ == "__main__":
    while True:
        print("\nSelect an `option:")
        print("1. Capture training images")
        print("2. Train model")
        print("3. Recognize face from live video and record attendance")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            person_name = input("Enter your name: ")
            capture_training_images(person_name)
        elif choice == "2":
            train_faces()
        elif choice == "3":
            recognize_face_live()
        elif choice == "4":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")