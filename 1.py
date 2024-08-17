import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os
from tensorflow.keras.preprocessing import image

# Load models
alz_model = load_model(r'C:\Users\HARSHIT\Documents\Summer-Break1\RPC Project\Frontend\saved model\alz_classifier.h5')
tumor_model = load_model(r'C:\Users\HARSHIT\Documents\Summer-Break1\RPC Project\Frontend\saved model\braintumor_new (1).h5')
pneumonia_model = pickle.load(open(r"C:\Users\HARSHIT\Documents\Summer-Break1\RPC Project\Frontend\saved model\pneumonia_model.sav", 'rb'))

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
    font-weigth: bold;
    
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
            
          <h2>Detection Wizard</h2>
               
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

# # Streamlit interface
st.sidebar.title("Detection Wizard")
st.sidebar.image('Images\sidebar_image.png', width=250)
option = st.sidebar.selectbox(
    "Select Operation:",
    ("Alzheimer's Detection", "Brain Tumor Detection", "Pneumonia Detection")
)

# Function to preprocess image for Alzheimer's model
def preprocess_image_for_alzheimer(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Tumor model
def preprocess_image_for_tumor(img):
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)
    return img_array

# Function to preprocess image for Pneumonia model
def preprocess_image_for_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(120, 120))  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Alzheimer’s Detection
if option == "Alzheimer's Detection":

    col1, col2 = st.columns([4,1.5])

    with col2:
        memory = Image.open('Images/m3.jpg')
        st.image(memory,width=250)

    with col1:
        st.title("Alzheimer's Classification")
        st.write("Upload an MRI image to check for Alzheimer's disease.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="alz_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_alzheimer(img)
        prediction = alz_model.predict(img_array)
        index = prediction.argmax()
        if index == 0:
            st.write("Most probably you have mild Alzheimer's disease.")
        elif index == 1:
            st.write("Most probably you have moderate Alzheimer's disease.")
        elif index == 2:
            st.write("Most probably you don't have Alzheimer's disease.")
        elif index == 3:
            st.write("Most probably you have very mild Alzheimer's disease.")

# Brain Tumor Detection
elif option == "Brain Tumor Detection":

    col1, col2 = st.columns([5,1])

    with col2:
        brain_tumor = Image.open('Images/brain.jpg')
        st.image(brain_tumor,width=150)

    with col1:
        st.title("Brain Tumor Classification")
        st.write("Upload an MRI image to check if you have a brain tumor.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg", key="tumor_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="RGB", caption="Uploaded MRI Image", use_column_width=True)
        img_array = preprocess_image_for_tumor(img)
        prediction = tumor_model.predict(img_array)
        index = prediction.argmax()
        if index != 2:
            st.write("You have a Tumor.")
            if index == 0:
                st.write("Most probably you have Glioma Tumor.")
            elif index == 1:
                st.write("Most probably you have Meningioma Tumor.")
            elif index == 3:
                st.write("Most probably you have Pituitary Tumor.")
        else:
            st.write("You do not have a Tumor.")

# Pneumonia Detection
elif option == "Pneumonia Detection":

    col1, col2 = st.columns([5, 1])  # Adjust the ratio of columns as needed

    with col2:
        lungs = Image.open('Images/lungs1.jpg')
        st.image(lungs, width=150)

    with col1:    
        st.title("Pneumonia Detection")
        st.write("Upload a chest X-ray image to detect Pneumonia.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"], key="pneumonia_uploader")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        temp_file_path = os.path.join("tempDir", uploaded_file.name)
        
        # Process the file (e.g., save it locally)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        preprocessed_image = preprocess_image_for_pneumonia(temp_file_path)

        col1, col2, col3 = st.columns([3, 1, 3])

        with col2:
            Proceed = st.button('Proceed')

        if Proceed:
            pneu_prediction = pneumonia_model.predict(preprocessed_image)
            predicted_class = 'Pneumonia' if pneu_prediction[0] > 0.75 else 'Normal'
            st.write(predicted_class)

            # Code for prediction
            pneu_diagnosis = 'I’m pleased to inform you that you do not have pneumonia.' if predicted_class == 'Normal' else 'It looks like you have pneumonia, which is an infection in your lungs.'
            st.success(pneu_diagnosis)
