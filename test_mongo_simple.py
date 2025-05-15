from pymongo import MongoClient

# Direct connection test without environment variables
uri = "mongodb+srv://admin:admin123@attendance.zkby0dr.mongodb.net/?retryWrites=true&w=majority&appName=attendance"

try:
    # Create a new client and connect to the server
    client = MongoClient(uri)
    
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    
    # Try to access the database and collection
    db = client['attendance_db']
    collection = db['attendance_records']
    
    # Try to insert a test document
    test_doc = {"test": "connection"}
    result = collection.insert_one(test_doc)
    print("Successfully inserted test document!")
    
    # Clean up
    collection.delete_one({"_id": result.inserted_id})
    print("Successfully cleaned up test document!")
    
except Exception as e:
    print(f"Error connecting to MongoDB: {e}") 