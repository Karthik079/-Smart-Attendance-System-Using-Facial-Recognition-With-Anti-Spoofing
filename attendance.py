import os
import csv
import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

# Debug environment variables
def debug_env_vars():
    mongo_uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('DATABASE_NAME')
    collection_name = os.getenv('COLLECTION_NAME')
    print(f"Loading .env from: {env_path}")
    print(f"MongoDB URI: {mongo_uri}")
    print(f"Database Name: {db_name}")
    print(f"Collection Name: {collection_name}")

# MongoDB connection details
MONGODB_URI = "mongodb+srv://admin:admin123@attendance.zkby0dr.mongodb.net/?retryWrites=true&w=majority&appName=attendance"
DATABASE_NAME = "attendance_db"
COLLECTION_NAME = "attendance_records"

# MongoDB connection setup
def get_mongodb_connection():
    try:
        client = MongoClient(MONGODB_URI)
        # Test the connection
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

def sync_to_mongodb(log_file):
    try:
        collection = get_mongodb_connection()
        
        # Read the CSV file
        with open(log_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            records = list(csv_reader)
            
            if not records:
                print("No records found in CSV file.")
                return
            
            # Update MongoDB
            for record in records:
                # Use timestamp and name as a unique identifier
                query = {
                    "Timestamp": record["Timestamp"],
                    "Name": record["Name"]
                }
                # Upsert the record (insert if not exists, update if exists)
                result = collection.update_one(query, {"$set": record}, upsert=True)
                if result.modified_count > 0:
                    print(f"Updated record for {record['Name']} at {record['Timestamp']}")
                elif result.upserted_id:
                    print(f"Inserted new record for {record['Name']} at {record['Timestamp']}")
                else:
                    print(f"Record already exists for {record['Name']} at {record['Timestamp']}")

    except Exception as e:
        print(f"Error in sync_to_mongodb: {str(e)}")
        raise

def record_attendance(name, log_file="data/attendance_log.csv"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to write header
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, "a", newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Name"])  # Write header if file is new
        writer.writerow([timestamp, name])
    
    print(f"Attendance recorded for {name} at {timestamp}.")
    
    # Sync to MongoDB after recording
    try:
        sync_to_mongodb(log_file)
        print("Successfully synchronized with MongoDB.")
    except Exception as e:
        print(f"Error synchronizing with MongoDB: {str(e)}")
        print("Data was saved to CSV but could not be synced to MongoDB.")
