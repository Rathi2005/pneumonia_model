import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import pickle

pneumonia_model = pickle.load(open(r"C:\Users\HARSHIT\Documents\Summer-Break1\RPC Project\Frontend\saved model\pneumonia_model.sav",'rb'))

############  Welcome Message  ############
import time

# Define the CSS styles as a string
css = """
<style>
body {
    background-color: '#FF0000';
}

h2 {
    color: rgb(56, 173, 177);
    text-align: center;
}

.custom-paragraph {
    font-size: 18px;
    text-align: center;
    color: #333;
}

.custom-welcome{
    # background-color: #D3D3D3;
    padding: 10px;
    # border: 3px solid black;
    border-radius: 10px;
    margin-top: 20%;
    font-family: 'Comic Sans MS';
}

.custom-welcome h2{
    font-family: 'Comic Sans MS';
}

</style>
"""

# Inject the CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Create a placeholder for the welcome message
welcome_message = st.empty()

if 'msg_displayed' not in st.session_state:          
    # session_state :-  is a special feature in Streamlit that allows you to keep track of variables and their values 
    #                   that persist as users interact with the app, even as the app is reloaded or refreshed.
    st.session_state.msg_displayed = False
    # .msg_displayed :- this creates a variable in the session_state and manage its value.

# Display the customized welcome message using the CSS styles
if not st.session_state.msg_displayed:
    welcome_message.markdown(
        """
        <div class='custom-welcome'>
            
          <h2>🎉 Welcome to the Amazing App! 🎉</h2>
          <p class='custom-paragraph'>We're glad to have you here.</p>
               
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.msg_displayed = True


# Wait for 5 seconds
time.sleep(3)

# Clear the welcome message
welcome_message.empty()

#####################################

# Add a title to the sidebar
st.sidebar.title("Health Scan")

st.sidebar.image('Images\sidebar_image.png',width=250  )

# Add a selectbox to the sidebar
option = st.sidebar.selectbox(
    "Select Operation:",
    ("Pneumonia Detection", "Scan X-ray", "Summarize Report", "Consult Doctor")
)

# Add a text input to the sidebar
user_input = st.sidebar.text_input("Enter your name:")

# Function to render the "Pneumonia Detection" page
def pneumonia_detection():
    st.title("Pneumonia Detection")
    st.write("Upload a chest X-ray image to detect Pneumonia.")

    col1, col2 = st.columns([5, 1])  # Adjust the ratio of columns as needed

    with col2:
        lungs = Image.open('Images/lungs1.jpg')
        st.image(lungs, width=200)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

        from tensorflow.keras.preprocessing import image

        def preprocess_image(img_path):
            img = image.load_img(img_path, target_size=(120, 120))  # Load and resize the image
            img_array = image.img_to_array(img)  # Convert the image to a numpy array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize the image
            return img_array

        if uploaded_file is not None:
            # Display the uploaded file

            st.image(uploaded_file, caption='Uploaded Image.',use_column_width=True)

            img = Image.open(uploaded_file)
            temp_file_path = os.path.join("tempDir", uploaded_file.name)

            # Process the file (e.g., save it locally)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            preprocessed_image = preprocess_image(temp_file_path)

            
            # creating button for prediction
            
            if st.button('Proceed'):
                pneu_prediction = pneumonia_model.predict(preprocessed_image)
                predicted_class = 'Pneumonia' if pneu_prediction[0] > 0.75 else 'Normal'
                st.write(predicted_class)

            # code for prediction
                pneu_diagnosis = 'I’m pleased to inform you that you do not have pneumonia.' if predicted_class=='Normal' else ' It looks like you have pneumonia, which is an infection in your lungs.'
                st.success(pneu_diagnosis)        

# Function to render the "Scan X-ray" page
def scan_xray():
    st.title("Scan X-ray")
    st.write("This page will handle X-ray scanning.")

# Function to render the "Summarize Report" page
def summarize_report():
    st.title("Consult Doctor")

    st.write("This page will handle doctor consultation.")

def consult_doctor():
    st.title("Consult Doctor")
    
    from datetime import datetime

    # Title of the app
    st.title("Doctor Appointment Scheduler")

    # Introduction
    st.write("Please fill out the form below to schedule an appointment with the doctor.")

    # Creating the form
    with st.form("appointment_form"):
        # Input fields for the form
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        appointment_date = st.date_input("Preferred Appointment Date")
        appointment_time = st.time_input("Preferred Appointment Time")
        notes = st.text_area("Additional Notes (Optional)")

        # Submit button
        submitted = st.form_submit_button("Schedule Appointment")

        # Processing the form data after submission
        if submitted:
            # Example: You can save the data to a database, send an email, etc.
            st.success(f"Thank you, {name}! Your appointment is scheduled for {appointment_date} at {appointment_time}. We will contact you at {email} or {phone} for confirmation.")
            st.write(f"Additional Notes: {notes}")

    # Footer
    st.write("We look forward to seeing you!")

# Conditional rendering based on the selected option
if option == "Pneumonia Detection":
    pneumonia_detection()
elif option == "Scan X-ray":
    scan_xray()
elif option == "Summarize Report":
    summarize_report()
elif option == "Consult Doctor":
    consult_doctor()
