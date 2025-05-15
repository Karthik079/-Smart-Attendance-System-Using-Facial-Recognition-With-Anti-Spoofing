from attendance import get_mongodb_connection

def test_connection():
    try:
        collection = get_mongodb_connection()
        print("MongoDB connection test successful!")
        # Try to insert a test document
        result = collection.insert_one({"test": "connection"})
        print("Test document inserted successfully!")
        # Clean up the test document
        collection.delete_one({"_id": result.inserted_id})
        print("Test document cleaned up successfully!")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")

if __name__ == "__main__":
    test_connection() 