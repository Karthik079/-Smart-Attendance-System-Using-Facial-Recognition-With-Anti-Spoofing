import os
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import pickle
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingRetrieval:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # MongoDB connection details
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.database_name = os.getenv('DATABASE_NAME')
        self.collection_name = os.getenv('COLLECTION_NAME', 'face_embeddings')
        
        # Initialize MongoDB connection
        self.client = None
        self.db = None
        self.collection = None
        
        # Cache for embeddings
        self.embeddings_cache = {}
        self.last_cache_update = None
        self.cache_duration = timedelta(minutes=5)  # Cache duration of 5 minutes
        
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

    def deserialize_embeddings(self, binary_data):
        """Deserialize Binary format back to numpy array"""
        return pickle.loads(binary_data)

    def get_all_embeddings(self, force_refresh=False):
        """
        Get all embeddings from MongoDB with caching
        Args:
            force_refresh: If True, force a cache refresh
        Returns:
            dict: Dictionary mapping user_ids to their embeddings
        """
        current_time = datetime.now()
        
        # Return cached embeddings if they're still valid
        if not force_refresh and self.last_cache_update is not None:
            if current_time - self.last_cache_update < self.cache_duration:
                return self.embeddings_cache

        try:
            self.connect_to_mongodb()
            
            # Clear existing cache
            self.embeddings_cache = {}
            
            # Fetch all documents from MongoDB
            cursor = self.collection.find({})
            
            # Process each document
            for doc in cursor:
                try:
                    user_id = doc['user_id']
                    embeddings = self.deserialize_embeddings(doc['embeddings'])
                    
                    # Store embeddings in cache
                    self.embeddings_cache[user_id] = np.array(embeddings)
                except KeyError as e:
                    logger.error(f"Invalid document format for user {doc.get('user_id', 'unknown')}: {str(e)}")
                    continue
            
            self.last_cache_update = current_time
            logger.info(f"Successfully retrieved embeddings for {len(self.embeddings_cache)} users")
            
            return self.embeddings_cache
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {str(e)}")
            raise
        finally:
            self.disconnect_from_mongodb()

    def get_user_embedding(self, user_id):
        """
        Get embeddings for a specific user
        Args:
            user_id: The ID of the user
        Returns:
            numpy.ndarray: Array of embeddings for the user
        """
        try:
            self.connect_to_mongodb()
            
            # Fetch document for the specific user
            doc = self.collection.find_one({'user_id': user_id})
            
            if doc and 'embeddings' in doc:
                embeddings = self.deserialize_embeddings(doc['embeddings'])
                return np.array(embeddings)
            else:
                logger.warning(f"No embeddings found for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings for user {user_id}: {str(e)}")
            raise
        finally:
            self.disconnect_from_mongodb() 