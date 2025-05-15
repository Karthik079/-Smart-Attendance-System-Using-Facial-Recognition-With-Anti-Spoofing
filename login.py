import streamlit as st
import os
import json
import hashlib

# Set page config
st.set_page_config(
    page_title="Login - Face Recognition Attendance System",
    page_icon="🔒",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .stTitle {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Login container */
    .login-container {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 400px;
        margin: 0 auto;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        background-color: #f8f9fa;
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.5rem;
        color: #1e293b;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        color: #1e3a8a;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    
    /* Error message */
    .error-message {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-message {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def hash_password(password):
    """Hash the password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists("data/users.json"):
        with open("data/users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    os.makedirs("data", exist_ok=True)
    with open("data/users.json", "w") as f:
        json.dump(users, f)

def create_default_admin():
    """Create default admin user if no users exist"""
    users = load_users()
    if not users:
        users["admin"] = {
            "password": hash_password("admin123"),
            "role": "admin"
        }
        save_users(users)

def verify_login(username, password):
    """Verify user credentials"""
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return True, users[username]["role"]
    return False, None

# Create default admin if needed
create_default_admin()

# Title and welcome message
st.markdown('<h1 class="stTitle">🔒 Login</h1>', unsafe_allow_html=True)
st.markdown('<p class="welcome-message">Welcome to the Face Recognition Attendance System</p>', unsafe_allow_html=True)

# Login form container
st.markdown('<div class="login-container">', unsafe_allow_html=True)

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username and password:
        success, role = verify_login(username, password)
        if success:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.markdown('<div class="success-message">Login successful! Redirecting...</div>', unsafe_allow_html=True)
            st.switch_page("pages/1_Home.py")
        else:
            st.markdown('<div class="error-message">Invalid username or password</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-message">Please enter both username and password</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Default credentials
st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #6b7280;">
        <p>Default credentials:</p>
        <p>Username: admin</p>
        <p>Password: admin123</p>
    </div>
""", unsafe_allow_html=True) 