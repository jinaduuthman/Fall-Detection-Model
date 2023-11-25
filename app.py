import streamlit as st
import pandas as pd
import joblib

# Load the trained model (ensure it's in the same directory or provide the full path)
rf_model = joblib.load('random_forest_model.pkl')

def predict_outcome(features):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([features], columns=['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'])
    # Predict
    prediction = rf_model.predict(input_data)
    return prediction

# Streamlit app layout
st.title('Fall Detection Prediction')

# Input fields for features
ax = st.number_input('AX', format="%.6f")
ay = st.number_input('AY', format="%.6f")
az = st.number_input('AZ', format="%.6f")
gx = st.number_input('GX', format="%.6f")
gy = st.number_input('GY', format="%.6f")
gz = st.number_input('GZ', format="%.6f")

# Button to make prediction
if st.button('Predict'):
    prediction = predict_outcome([ax, ay, az, gx, gy, gz])
    # if prediction[0]:
    #     st.markdown(f"<h2 style='color: red;'>Prediction: Fall Detected</h2>", unsafe_allow_html=True)
    # else:
    #     st.markdown(f"<h2 style='color: green;'>Prediction: No Fall Detected</h2>", unsafe_allow_html=True)

    # Using an expander to simulate a modal
    with st.expander("See Prediction Result", expanded=True):
        if prediction[0]:
            # Fall Detected
            st.markdown(f"<div style='background-color:lightcoral; padding: 10px; border-radius: 5px;'> <h2 style='color: white;'>Fall Detected, Dialing 911....</h2></div>", unsafe_allow_html=True)
        else:
            # No Fall Detected
            st.markdown(f"<div style='background-color:lightgreen; padding: 10px; border-radius: 5px;'><h2 style='color: white;'>No Fall Detected</h2></div>", unsafe_allow_html=True)
    # st.write(f'Prediction: {"Fall Detected" if prediction[0] else "No Fall Detected"}')

# Run the Streamlit app by navigating to the app's directory and running 'streamlit run app.py'
