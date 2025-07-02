import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from home_page_detail import home_page
from about_cdss import about_cdss 
from streamlit_option_menu import option_menu
from data_utils import load_data, preprocess_data
from model_utils import load_model as load_model_utils, predict
from visualization import plot_results
from ui_components import custom_sidebar
from config import DATA_PATH, MODEL_PATH
from disease_info import get_disease_info, disease_info

# Load trained RNN model
try:
    rnn_model = load_model('CDSS_RNN.h5', compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please check TensorFlow/Keras version compatibility")
# Load features and class labels
features = pd.read_csv('data/features.csv')
classes = pd.read_csv('data/classes.csv')

num_features = len(features) - 1

# Prepare encoder for class labels
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(classes.values.reshape(-1, 1))

# Set page configuration
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",  
)

st.markdown("""
    <style>
        /* Remove top and bottom padding */
        .block-container {
            padding-top: 4rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# Store number of features in session state for use in about_cdss page
if 'num_features' not in st.session_state:
    st.session_state.num_features = num_features

# Place your disease_info dictionary here
# ===========================================
# disease_info = {
#     
# }
# ===========================================

def predict_symptoms(input_data):
    reshaped_data = input_data.values.reshape((1, 1, input_data.shape[1]))
    pred_probs = rnn_model.predict(reshaped_data)
    
    pred_classes = np.argmax(pred_probs, axis=1)
    predicted_labels = encoder.inverse_transform(np.eye(pred_probs.shape[1])[pred_classes])
    
    accuracy = np.max(pred_probs)
    
    return predicted_labels[0][0], accuracy

def display_disease_info(prediction):
    info = get_disease_info(prediction)
    st.title(f"{prediction} - Disease Overview")
    if info is None:
        st.warning("No information available for this disease.")
        return
    tabs = st.tabs(["Overview", "Tests & Treatment", "Safety Measures", "Specialist Contact Details"])
    with tabs[0]:
        # Show description if available
        description = info.get('description', None)
        if description:
            st.markdown(f"**Description:** {description}")
        # Show symptoms
        st.subheader("Common Symptoms")
        symptoms = info.get('Symptoms', [])
        if symptoms:
            for item in symptoms:
                st.markdown(f"- {item}")
        else:
            st.warning("No symptoms available.")
        # Show consulting specialist
        specialist = info.get('Consulting Specialist', None)
        if specialist:
            st.subheader("Consulting Specialist")
            if isinstance(specialist, list):
                for s in specialist:
                    st.markdown(f"- {s}")
            else:
                st.markdown(f"- {specialist}")
        # Show resource link
        resource = info.get('Resource Link', None)
        if resource:
            st.markdown(f"[More Information]({resource})")

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Consulting Specialist")
            specialist = info.get('Consulting Specialist', [])
            if specialist:
                for item in specialist:
                    st.markdown(f"- {item}")
            else:
                st.warning("No specialist available.")
        with col2:
            st.subheader("Suggested Tests")
            tests = info.get('Diagnostic Tests', [])
        
            if tests:
                for item in tests:
                    st.markdown(f"- {item}")
            else:
                st.warning("No tests available.")

        st.subheader("Common Treatments")
        treatments = info.get('Treatments', [])
        
        if treatments:
            for item in treatments:
                st.markdown(f"- {item}")
        else:
            st.warning("No treatments available.")

    with tabs[2]:
        st.subheader("Precautions")
        precautions = info.get('Precautions', [])
        
        if precautions:
            for item in precautions:
                st.markdown(f"⚠️ {item}")
        else:
            st.warning("No precautions available.")

    with tabs[3]:
        st.write(f"🔗 To Learn More about {prediction}: [Click Here]({info.get('Resource Link', '#')})")
        for line in info.get('Specialist', []):
            st.write(f"**{line}**")

# Streamlit app
def main():
    selected = option_menu(
            menu_title=None, 
            options=["Home", "About CDSS", "Use Tool"],  
            icons=["house", "info-circle", "gear"],  
            menu_icon="cast",  
            default_index=0, 
            orientation="horizontal",
        )

    if selected == "Home":
        home_page()
        st.markdown("---")
        st.markdown("### Get Started")
        st.markdown("Navigate to the **Use Tool** page to start predicting diseases based on symptoms.")

        st.markdown("""
        <hr style='border:1px solid gray'>
        <p style='text-align: center;'>© 2025 CDSS | Contact: <a href="mailto:prithivirajdeivendhran01@gmail.com" style="text-decoration: none; color: gray;">prithivirajdeivendhran01@gmail.com</a></p>
        """, unsafe_allow_html=True)

    elif selected == "About CDSS":
        about_cdss()

    elif selected == "Use Tool":
        st.write("Please select your symptoms below:")

        symptom_list = np.array(features[:-1]).flatten().tolist()
        selected_symptoms = st.multiselect("Symptoms:", symptom_list, default=[])

        input_data = pd.DataFrame(np.zeros((1, num_features)), columns=symptom_list)

        for symptom in selected_symptoms:
            input_data[symptom] = 1

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                disease, accuracy = predict_symptoms(input_data)
                st.success(f"""
                        **Predicted Disease:** {disease} 
                        
                        **Level of Confidence:** {accuracy:.2%}
                            
                        """)
                display_disease_info(disease)
            st.markdown("""
            <hr style='border:1px solid gray'>
            <p style='text-align: center;'>© 2025 CDSS | Contact: <a href="mailto:prithivirajdeivendhran01@gmail.com" style="text-decoration: none; color: gray;">prithivirajdeivendhran01@Gmail.com</a></p>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()