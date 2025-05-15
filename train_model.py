import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle
from datetime import datetime
from bson.binary import Binary
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceEmbeddingTrainer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # MongoDB connection details
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.database_name = os.getenv('DATABASE_NAME')
        self.collection_name = os.getenv('COLLECTION_NAME', 'face_embeddings')
        
        if not self.mongodb_uri or not self.database_name:
            raise ValueError("MongoDB configuration missing. Please check your .env file.")
        
        # Initialize MongoDB connection
        self.client = None
        self.db = None
        self.collection = None
        
        # Initialize FaceNet model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=20, 
            keep_all=False, 
            post_process=True,
            device=self.device
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Directory paths
        self.dataset_dir = "data/training_images"
        
        logger.info(f"Initialized trainer with device: {self.device}")
        
    def connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
            if self.client is None:
                self.client = MongoClient(self.mongodb_uri)
                self.db = self.client[self.database_name]
                self.collection = self.db[self.collection_name]
                logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def disconnect_from_mongodb(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            logger.info("Disconnected from MongoDB")

    def serialize_embeddings(self, embeddings):
        """Serialize list of numpy array embeddings to Binary format for MongoDB storage"""
        return Binary(pickle.dumps(embeddings, protocol=2))

    def normalize_embedding(self, embedding):
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def extract_face_embedding(self, image_path):
        """Extracts FaceNet embedding from an image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Error: Could not load {image_path}")
                return None
            
            # Convert to RGB and normalize pixel values
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect and align face
            face = self.mtcnn(img_rgb)
            if face is None:
                logger.warning(f"No face detected in {image_path}")
                return None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.facenet(face.unsqueeze(0))
            
            # Move to CPU and convert to numpy
            embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize the embedding
            embedding = self.normalize_embedding(embedding)
            
            # Validate embedding
            if not np.all(np.isfinite(embedding)):
                logger.warning(f"Invalid embedding generated for {image_path}")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def store_person_embeddings(self, user_id, embeddings, image_paths):
        """Store all embeddings for a person in a single document"""
        try:
            # Validate embeddings
            if not all(isinstance(emb, np.ndarray) for emb in embeddings):
                raise ValueError("All embeddings must be numpy arrays")
            
            if not all(np.all(np.isfinite(emb)) for emb in embeddings):
                raise ValueError("All embeddings must contain valid values")
            
            # Verify embedding sizes
            if not all(emb.shape[0] == 512 for emb in embeddings):
                raise ValueError("All embeddings must be 512-dimensional")
            
            # Create metadata for each embedding
            embeddings_metadata = [
                {
                    'source_image': os.path.basename(img_path),
                    'embedding_norm': float(np.linalg.norm(emb))
                }
                for emb, img_path in zip(embeddings, image_paths)
            ]
            
            document = {
                'user_id': user_id,
                'embeddings': self.serialize_embeddings(embeddings),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'metadata': {
                    'device': str(self.device),
                    'embedding_count': len(embeddings),
                    'embeddings_info': embeddings_metadata
                }
            }
            
            # Update or insert the document
            result = self.collection.update_one(
                {'user_id': user_id},
                {'$set': document},
                upsert=True
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated embeddings for {user_id} with {len(embeddings)} new embeddings")
            else:
                logger.info(f"Created new document for {user_id} with {len(embeddings)} embeddings")
            
            return result.upserted_id or result.modified_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings for {user_id}: {str(e)}")
            raise

    def train_faces(self, force_retrain=False):
        """Train faces and store embeddings directly in MongoDB."""
        total_processed = 0
        total_users = 0
        
        try:
            self.connect_to_mongodb()
            
            # Get list of training directories
            person_dirs = [d for d in os.listdir(self.dataset_dir) 
                         if os.path.isdir(os.path.join(self.dataset_dir, d))]
            
            if not person_dirs:
                logger.error(f"No training directories found in {self.dataset_dir}")
                return
            
            logger.info(f"Found {len(person_dirs)} persons to process")
            
            for person_name in person_dirs:
                person_path = os.path.join(self.dataset_dir, person_name)
                
                # Check if person already exists in MongoDB
                existing_doc = self.collection.find_one({'user_id': person_name})
                if existing_doc and not force_retrain:
                    logger.info(f"Skipping {person_name}, already has embeddings stored")
                    continue
                elif existing_doc and force_retrain:
                    logger.info(f"Retraining {person_name}")
                
                # Process all images for the person
                image_files = [f for f in os.listdir(person_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    logger.warning(f"No valid images found for {person_name}")
                    continue
                
                logger.info(f"Processing {len(image_files)} images for {person_name}")
                
                successful_embeddings = []
                successful_image_paths = []
                failed_count = 0
                
                for image_file in image_files:
                    image_path = os.path.join(person_path, image_file)
                    embedding = self.extract_face_embedding(image_path)
                    
                    if embedding is not None:
                        successful_embeddings.append(embedding)
                        successful_image_paths.append(image_path)
                    else:
                        failed_count += 1
                
                if successful_embeddings:
                    try:
                        self.store_person_embeddings(person_name, successful_embeddings, successful_image_paths)
                        total_processed += len(successful_embeddings)
                        total_users += 1
                        logger.info(f"Completed {person_name}: {len(successful_embeddings)} successful, {failed_count} failed")
                    except Exception as e:
                        logger.error(f"Failed to store embeddings for {person_name}: {str(e)}")
                else:
                    logger.warning(f"No successful embeddings for {person_name}")
            
            logger.info(f"Training complete. Processed {total_processed} embeddings for {total_users} users")
            
        except Exception as e:
            logger.error(f"Training process failed: {str(e)}")
            raise
        finally:
            self.disconnect_from_mongodb()

def train_faces(force_retrain=False):
    """
    Standalone function to train face embeddings.
    Args:
        force_retrain: If True, retrain existing users
    """
    trainer = FaceEmbeddingTrainer()
    try:
        trainer.train_faces(force_retrain=force_retrain)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def main():
    try:
        train_faces(force_retrain=True)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

