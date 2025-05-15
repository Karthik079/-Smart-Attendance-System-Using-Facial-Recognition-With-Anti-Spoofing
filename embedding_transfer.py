import os
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle
import logging
from datetime import datetime
from bson.binary import Binary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingTransfer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # MongoDB connection details
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.database_name = os.getenv('DATABASE_NAME')
        self.collection_name = 'face_embeddings'  # Specific collection for embeddings
        
        # Initialize MongoDB connection
        self.client = None
        self.db = None
        self.collection = None
    
    def connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
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
            logger.info("Disconnected from MongoDB")

    def serialize_embedding(self, embedding):
        """Serialize numpy array embedding to Binary format for MongoDB storage"""
        return Binary(pickle.dumps(embedding, protocol=2))

    def deserialize_embedding(self, binary_data):
        """Deserialize Binary format back to numpy array"""
        return pickle.loads(binary_data)

    def transfer_embedding(self, user_id, embedding, metadata=None):
        """
        Transfer a single embedding to MongoDB
        Args:
            user_id: Identifier for the user
            embedding: numpy array of face embedding
            metadata: Additional metadata (optional)
        """
        try:
            if metadata is None:
                metadata = {}
            
            document = {
                'user_id': user_id,
                'embedding': self.serialize_embedding(embedding),
                'created_at': datetime.utcnow(),
                'metadata': metadata
            }
            
            result = self.collection.insert_one(document)
            logger.info(f"Successfully transferred embedding for user {user_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to transfer embedding for user {user_id}: {str(e)}")
            raise

    def transfer_all_embeddings(self, embeddings_dir):
        """
        Transfer all embeddings from a directory to MongoDB
        Args:
            embeddings_dir: Directory containing embedding files
        """
        try:
            self.connect_to_mongodb()
            
            successful_transfers = 0
            failed_transfers = 0
            
            for filename in os.listdir(embeddings_dir):
                if filename.endswith('.npz'):
                    try:
                        # Extract user_id from filename (removing .npz extension)
                        user_id = filename[:-4]  # Remove .npz extension
                        
                        # Load embedding
                        embedding_path = os.path.join(embeddings_dir, filename)
                        npz_data = np.load(embedding_path)
                        
                        # NPZ files are dictionaries of arrays. We'll transfer all arrays
                        for key in npz_data.files:
                            embedding = npz_data[key]
                            
                            # Transfer to MongoDB with metadata including the key name
                            self.transfer_embedding(
                                user_id=user_id,
                                embedding=embedding,
                                metadata={
                                    'source_file': filename,
                                    'array_key': key
                                }
                            )
                        
                        successful_transfers += 1
                        logger.info(f"Processed file {filename} with {len(npz_data.files)} embeddings")
                        
                    except Exception as e:
                        logger.error(f"Failed to transfer {filename}: {str(e)}")
                        failed_transfers += 1
                        
            logger.info(f"Transfer complete. Successful files: {successful_transfers}, Failed files: {failed_transfers}")
            
        finally:
            self.disconnect_from_mongodb()

def main():
    """Example usage of the EmbeddingTransfer class"""
    embeddings_dir = "data/embeddings"  # Update this path as needed
    
    transfer = EmbeddingTransfer()
    try:
        transfer.transfer_all_embeddings(embeddings_dir)
    except Exception as e:
        logger.error(f"Transfer process failed: {str(e)}")

if __name__ == "__main__":
    main() 