import re
from flask import Flask, get_flashed_messages, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mail import Mail, Message
import numpy as np
import logging
import json
import pyrebase
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask_socketio import SocketIO, send
from datetime import datetime, timedelta
import threading
from email_validator import validate_email, EmailNotValidError
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates")
app.secret_key = 'v1i2s3h4a5l6@y1a2d3a4v5@#$%'
socketio = SocketIO(app)

# Initialize Firebase Admin SDK for Firestore
try:
    cred = credentials.Certificate("C:\\Users\\VISHAL YADAV\\vish\\Disease\\key.json")  # Replace with your service account key file
    firebase_admin.initialize_app(cred)
    db = firestore.client()  # Firestore client
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Firebase Admin SDK: {e}")

# Firebase Configuration for Pyrebase (Authentication)
firebase_config = {
    "apiKey": "AIzaSyB3jKn1QyLtnieKZzAqAtBIFXaiwNVtnG8",
    "authDomain": "disease-prediction-5bedb.firebaseapp.com",
    "databaseURL": "https://disease-prediction-5bedb-default-rtdb.firebaseio.com",
    "projectId": "disease-prediction-5bedb",
    "storageBucket": "disease-prediction-5bedb.appspot.com",
    "messagingSenderId": "153602824873",
    "appId": "1:153602824873:web:5af89993f0166c4db167b8"
}

# Initialize Pyrebase for Authentication
firebase = pyrebase.initialize_app(firebase_config)
auth_pyrebase = firebase.auth()  # Pyrebase authentication

# Load training dataset
train_df = pd.read_csv("training.csv")  # Replace with actual file name
X_train = train_df.drop(columns=["prognosis"])  # Features (Symptoms)
y_train = train_df["prognosis"]  # Target (Disease)

# Load testing dataset
test_df = pd.read_csv("testing.csv")  # Replace with actual file name
X_test = test_df.drop(columns=["prognosis"])
y_test = test_df["prognosis"]

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Get available symptoms from dataset
symptom_columns = list(X_train.columns)

# Load disease information
with open('disease_info.json', 'r') as f:
    disease_info = json.load(f)

# Store messages temporarily (in-memory, replace with a database for production)
messages = []

# Function to delete messages after 24 hours
def delete_messages_after_24_hours():
    while True:
        global messages
        current_time = datetime.now()
        messages = [msg for msg in messages if current_time - msg['timestamp'] < timedelta(hours=24)]
        threading.Event().wait(3600)  # Check every hour

# Start the message deletion thread
threading.Thread(target=delete_messages_after_24_hours, daemon=True).start()

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-email-password'
mail = Mail(app)

# Email Validation
def is_valid_email(email):
    try:
        # Validate the email
        valid = validate_email(email)
        # Update the email with the normalized form
        email = valid.email
        return True
    except EmailNotValidError as e:
        # Email is not valid
        return False

# Password Validation
def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login first to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Define the admin email
ADMIN_EMAIL = 'vishbha9324@gmail.com'  # Replace with your actual admin email

# Admin Required Decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login first to access this page.', 'error')
            return redirect(url_for('login'))
        
        # Fetch user email from session
        user_email = session.get('user', {}).get('email')
        
        # Check if the user's email matches the admin email
        if user_email != ADMIN_EMAIL:
            flash('You do not have permission to access this page.', 'error')
            return redirect(url_for('home'))
        
        return f(*args, **kwargs)
    return decorated_function

# Set admin role for a user based on email
def set_admin_by_email(email):
    try:
        # Fetch the user by email
        user = auth.get_user_by_email(email)
        
        # Set custom claim 'admin: True' for the user
        auth.set_custom_user_claims(user.uid, {"admin": True})
        print(f"Admin role set for user: {user.uid} ({email})")
    except Exception as e:
        print(f"Error setting admin role: {e}")

# Set admin role for the designated admin email
set_admin_by_email(ADMIN_EMAIL)

# Loading Page
@app.route('/loading')
def loading():
    return render_template('loading.html')

# Home Page (Requires Login)
@app.route('/')
def home():
    return render_template('home.html')

# About Page (No Login Required)
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page (No Login Required)
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Socket.IO for Real-Time Communication
@socketio.on('message')
def handle_message(msg):
    timestamp = datetime.now()
    messages.append({'text': msg, 'timestamp': timestamp})
    send(msg, broadcast=True)

from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = 'user'  # Default role is 'user'

        # Validate email
        if not is_valid_email(email):
            flash('Invalid email address. Please provide a valid email.', 'danger')
            return redirect(url_for('register'))

        # Validate password
        if password != confirm_password:
            flash('Passwords do not match. Please re-enter your password.', 'danger')
            return redirect(url_for('register'))

        if not is_strong_password(password):
            flash('Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character.', 'danger')
            return redirect(url_for('register'))

        try:
            # Hash the password before storing it
            hashed_password = generate_password_hash(password)  # Remove method='sha256'

            # Check if the email already exists in Firestore
            users_ref = db.collection('users').where('email', '==', email).stream()
            if any(users_ref):
                flash('This email is already registered. Please use a different email or log in.', 'danger')
                return redirect(url_for('register'))

            # Create user in Firebase Authentication
            user = auth_pyrebase.create_user_with_email_and_password(email, password)
            print(f"User created in Firebase Authentication: {user['localId']}")

            # Store additional user data in Firestore
            user_data = {
                'name': name,
                'email': email,
                'uid': user['localId'],  # Use 'localId' instead of 'uid'
                'role': role,  # Default role is 'user'
                'password': hashed_password  # Store hashed password
            }

            # Debugging: Print user data to be stored
            print(f"User data to be stored: {user_data}")

            # Save user data to Firestore
            db.collection('users').document(user['localId']).set(user_data)
            print(f"User data stored in Firestore for UID: {user['localId']}")

            # Send confirmation email
            msg = Message('Registration Successful', sender='your-email@gmail.com', recipients=[email])
            msg.body = f'Thank you for registering with us, {name}!'
            mail.send(msg)

            flash('Registration successful! Please check your email for confirmation.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            # Handle Firebase Authentication errors
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                flash('This email is already registered. Please use a different email or log in.', 'danger')
            elif "WEAK_PASSWORD" in error_message:
                flash('The password is too weak. Please choose a stronger password.', 'danger')
            elif "INVALID_EMAIL" in error_message:
                flash('Invalid email address. Please provide a valid email.', 'danger')
            elif "MISSING_PASSWORD" in error_message:
                flash('Password is required. Please enter a password.', 'danger')
            elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                flash('Too many attempts. Please try again later.', 'danger')
            else:
                flash(f'An unexpected error occurred: {error_message}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return redirect(url_for('login'))

        try:
            # Verify user credentials using Firebase Authentication
            user = auth_pyrebase.sign_in_with_email_and_password(email, password)
            
            # Fetch user data from Firestore
            user_ref = db.collection('users').document(user['localId'])
            user_doc = user_ref.get()

            if user_doc.exists:
                user_data = user_doc.to_dict()
                # Verify the password
                if check_password_hash(user_data['password'], password):
                    # Store user session and ID token
                    session['user'] = user
                    session['id_token'] = user['idToken']
                    session['user_data'] = user_data
                    session['login_success'] = True  # Set session flag for login success
                    flash('Login successful!', 'success')
                    return redirect(url_for('login'))  # Redirect to login page to show modal
                else:
                    flash('Invalid email or password. Please try again.', 'danger')
            else:
                flash('User data not found. Please contact support.', 'danger')
                return redirect(url_for('login'))

        except auth_pyrebase.AuthError as e:
            # Handle Firebase Authentication errors
            error_message = 'Invalid email or password. Please try again.'
            if 'INVALID_PASSWORD' in str(e) or 'EMAIL_NOT_FOUND' in str(e):
                error_message = 'Invalid email or password. Please try again.'
            flash(error_message, 'danger')
            return redirect(url_for('login'))

        except Exception as e:
            # Handle other exceptions
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(url_for('login'))

    # Render the login page for GET requests
    return render_template('login.html')


# Forgot Password Page
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        # Server-side validation
        if not email:
            flash('Email is required.', 'error')
            return render_template('forgot_password.html')

        try:
            # Send password reset email using Firebase Authentication
            auth_pyrebase.send_password_reset_email(email)
            flash('Password reset email sent. Please check your inbox.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Error sending password reset email: {str(e)}")
            flash('An error occurred. Please try again.', 'error')

    return render_template('forgot_password.html')

# Dashboard
@app.route('/dashboard')
@login_required  # Ensure only logged-in users can access this route
def dashboard():
    user_id = session['user']['localId']  # Get the current user's ID
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()

    if user_data.get('role') == 'admin':
        # Fetch all users and patients for admin
        users_ref = db.collection('users').stream()
        users = [user.to_dict() for user in users_ref]
        
        patients_ref = db.collection('patients').stream()
        patients = [patient.to_dict() for patient in patients_ref]
        
        # Fetch all predictions for admin
        predictions_ref = db.collection('patient_history').stream()
        predictions = [pred.to_dict() for pred in predictions_ref]
        
        return render_template('dashboard.html', users=users, patients=patients, predictions=predictions, role='admin')
    else:
        # Fetch only the current user's predictions for regular users
        user_predictions_ref = db.collection('patient_history').document(user_id).collection('predictions').stream()
        predictions = [pred.to_dict() for pred in user_predictions_ref]
        
        return render_template('dashboard.html', predictions=predictions, role='user')

# Profile Page (Requires Login)
@app.route('/profile')
@login_required
def profile():
    if 'user' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    try:
        user_id = session['user']['localId']  # Use the correct user ID

        # Fetch user data from Firestore
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get().to_dict()

        if not user_data:
            flash('User data not found.', 'error')
            return redirect(url_for('login'))

        # Fetch patient form data from Firestore
        patient_ref = db.collection('patients').where('email', '==', user_data['email']).stream()
        patient_data = None
        for patient in patient_ref:
            patient_data = patient.to_dict()
            break  # Assuming there's only one patient record per user

        # Fetch recent activity (e.g., predictions, logins, etc.)
        activity_ref = db.collection('patient_history').document(user_id).collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5).stream()
        recent_activity = []
        for activity in activity_ref:
            data = activity.to_dict()
            recent_activity.append({
                'description': f"Predicted disease: {data.get('disease', 'Unknown')}",
                'timestamp': data.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S')
            })

        # Pass data to the template
        return render_template('profile.html',
                               user_data=user_data,
                               patient_data=patient_data,  # Pass patient_data here
                               recent_activity=recent_activity)  # Pass recent_activity here

    except Exception as e:
        logging.error(f"Error fetching profile data: {str(e)}")
        flash('An error occurred while fetching profile data.', 'error')
        return redirect(url_for('home'))

# Patient Form Page (Requires Login)
@app.route('/patient-form', methods=['GET', 'POST'])
@login_required
def patient_form():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        contact = request.form.get('contact')
        address = request.form.get('address')
        
        patient_data = {
            'name': name,
            'age': age,
            'gender': gender,
            'contact': contact,
            'address': address,
            'email': session['user']['email']
        }
        
        try:
            # Save patient data to Firestore
            db.collection('patients').add(patient_data)
            flash('Patient information saved successfully!', 'success')
            return redirect(url_for('symptom_selection'))
        except Exception as e:
            logging.error(f"Error saving patient information: {str(e)}")
            flash('Error saving patient information. Please try again.', 'error')
            
    return render_template('patient_form.html')

# Symptom Selection Page (Requires Login)
@app.route('/symptom_selection', methods=['GET', 'POST'])
@login_required
def symptom_selection():
    # Get the list of symptoms from the dataset
    symptoms = symptom_columns  # symptom_columns is already loaded from the dataset

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')  # Get selected symptoms from the dropdown
        if not selected_symptoms:
            flash('Please select at least one symptom', 'error')
            return render_template('symptom_selection.html', symptoms=symptoms)

        # Convert selected symptoms into model format
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        
        # Predict disease
        predicted_disease = model.predict([input_data])[0]
        
        # Get disease information
        disease_details = disease_info.get(predicted_disease, {
            "description": "No detailed information available.",
            "causes": "Information not available.",
            "treatment": ["Consult a healthcare professional"],
            "prevention": ["Consult a healthcare professional"]
        })
        
        # Save to patient history in Firestore
        try:
            user_id = session['user']['localId']  # Use the correct user ID
            history_data = {
                "symptoms": selected_symptoms,
                "disease": predicted_disease,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            db.collection('patient_history').document(user_id).collection('predictions').add(history_data)
        except Exception as e:
            logging.error(f"Error saving to Firestore: {str(e)}")
            flash('Prediction made but could not save to history', 'warning')
        
        return render_template('result.html',
                            disease=predicted_disease,
                            symptoms=selected_symptoms,
                            disease_info=disease_details)
    
    return render_template('symptom_selection.html', symptoms=symptoms)

# Predict Disease (API Endpoint) (Requires Login)
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get selected symptoms from the form
        selected_symptoms = request.form.getlist('symptoms')
        if not selected_symptoms:
            flash('Please select at least one symptom.', 'error')
            return jsonify({"error": "Please select at least one symptom!"}), 400

        # Convert selected symptoms into model format
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        
        # Predict disease
        predicted_disease = model.predict([input_data])[0]
        print(f"Predicted Disease: {predicted_disease}")  # Debugging

        # Get disease information
        disease_details = disease_info.get(predicted_disease, {
            "description": "No detailed information available.",
            "causes": "Information not available.",
            "treatment": ["Consult a healthcare professional"],
            "prevention": ["Consult a healthcare professional"]
        })

        # Save to patient history in Firestore
        try:
            user_id = session['user','name']['localId']  # Use the correct user ID
            history_data = {
                "symptoms": selected_symptoms,
                "disease": predicted_disease,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            # Add the prediction to the user's history
            db.collection('patient_history').document(user_id).collection('predictions').add(history_data)
            print(f"Data saved to Firestore: {history_data}")  # Debugging
        except Exception as e:
            logging.error(f"Error saving to Firestore: {str(e)}")
            flash('Prediction made but could not save to history.', 'warning')
            return jsonify({"warning": "Prediction made but could not save to history"}), 500

        # Return the prediction and disease details as JSON
        return jsonify({
            "predicted_disease": predicted_disease,
            "disease_info": disease_details,
            "message": "Prediction saved successfully!"
        })

    except Exception as e:
        logging.error(f"Error in /predict route: {str(e)}")
        flash('An error occurred while making the prediction. Please try again.', 'danger')
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500
# Patient History Page (Requires Login)
@app.route('/patient_history')
@login_required
def patient_history():
    try:
        # Get the current user's ID from the session
        user_id = session['user']['localId']

        # Fetch patient details (if needed)
        patients_ref = db.collection('patients').where('email', '==', session['user']['email']).stream()
        patient_details = [patient.to_dict() for patient in patients_ref]

        # Fetch prediction history from Firestore
        predictions_ref = db.collection('patient_history').document(user_id).collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        # Prepare the history data
        history = []
        for pred in predictions_ref:
            data = pred.to_dict()
            history.append({
                'symptoms': data.get('symptoms', []),
                'disease': data.get('disease', 'Unknown'),
                'timestamp': data.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S')  # Format timestamp
            })

        # Render the template with patient details and history
        return render_template('patient_history.html', patient_details=patient_details, history=history)

    except Exception as e:
        logging.error(f"Error retrieving patient history: {str(e)}")
        flash('Error retrieving patient history. Please try again.', 'error')
        return redirect(url_for('home'))
    
@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        feedback_message = request.form.get('feedbackInput')

        if feedback_message:
            # Save feedback to Firestore
            feedback_data = {
                'message': feedback_message,
                'timestamp': firestore.SERVER_TIMESTAMP  # Automatically adds the server timestamp
            }
            db.collection('feedback').add(feedback_data)  # Add feedback to the 'feedback' collection

            flash('Thank you for your feedback!', 'success')
        else:
            flash('Please enter your feedback.', 'error')

    return redirect(url_for('home'))


# Logout (Requires Login)
@app.route('/logout')
@login_required
def logout():
    session.pop('user', None)
    session.pop('user_data', None)
    session.pop('id_token', None)
     # Set the logout success flag
    session['logout_success'] = True
    flash('You have been logged out successfully!', 'success')  # Flash message for logout success
    return redirect(url_for('home'))


if __name__ == '__main__':
    try:
        socketio.run(app, debug=True)
    except Exception as e:
        logging.error(f"Application error: {str(e)}")