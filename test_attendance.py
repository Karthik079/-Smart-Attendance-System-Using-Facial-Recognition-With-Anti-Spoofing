from attendance import record_attendance

def test_attendance():
    try:
        # Record attendance for a test user
        record_attendance("Test User")
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_attendance() 