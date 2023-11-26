import streamlit as st
import pandas as pd
import joblib
from firebase import firebase
import streamlit as st
import time

# Load the trained model (ensure it's in the same directory or provide the full path)
rf_model = joblib.load('random_forest_model.pkl')

#Firebase settings config
firebase_url = 'https://falldetection-6c89a-default-rtdb.firebaseio.com/'
firebase_db = firebase.FirebaseApplication(firebase_url, None)

def predict_outcome(features):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([features], columns=['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'])
    # Predict
    prediction = rf_model.predict(input_data)
    return prediction

def get_sensor_data():
    # 从 Firebase 获取加速度计和陀螺仪数据
    accelerometer_data = firebase_db.get('/Accelerometer', None)
    gyroscope_data = firebase_db.get('/Gyroscope', None)
    
    return accelerometer_data, gyroscope_data

#THis version of display_data- is only used for the predict button.
def display_data_(accelerometer_data, gyroscope_data):
    # Extract the accelerometer and gyroscope data
    ax = accelerometer_data.get('x', 0) if accelerometer_data else 0
    ay = accelerometer_data.get('y', 0) if accelerometer_data else 0
    az = accelerometer_data.get('z', 0) if accelerometer_data else 0

    gx = gyroscope_data.get('x', 0) if gyroscope_data else 0
    gy = gyroscope_data.get('y', 0) if gyroscope_data else 0
    gz = gyroscope_data.get('z', 0) if gyroscope_data else 0

    # Display the accelerometer data
    st.subheader("Accelerometer Data:")
    st.metric(label="AX (Accelerometer X)", value=f"{ax:.6f}")
    st.metric(label="AY (Accelerometer Y)", value=f"{ay:.6f}")
    st.metric(label="AZ (Accelerometer Z)", value=f"{az:.6f}")

    # Display the gyroscope data
    st.subheader("Gyroscope Data:")
    st.metric(label="GX (Gyroscope X)", value=f"{gx:.6f}")
    st.metric(label="GY (Gyroscope Y)", value=f"{gy:.6f}")
    st.metric(label="GZ (Gyroscope Z)", value=f"{gz:.6f}")

    return ax, ay, az, gx, gy, gz



def display_data(accelerometer_data, gyroscope_data, column):
    # Extract the accelerometer and gyroscope data
    with column:
        ax = accelerometer_data.get('x', 0) if accelerometer_data else 0
        ay = accelerometer_data.get('y', 0) if accelerometer_data else 0
        az = accelerometer_data.get('z', 0) if accelerometer_data else 0

        gx = gyroscope_data.get('x', 0) if gyroscope_data else 0
        gy = gyroscope_data.get('y', 0) if gyroscope_data else 0
        gz = gyroscope_data.get('z', 0) if gyroscope_data else 0

        # Display the accelerometer data
        st.subheader("Accelerometer Data:")
        st.metric(label="AX (Accelerometer X)", value=f"{ax:.6f}")
        st.metric(label="AY (Accelerometer Y)", value=f"{ay:.6f}")
        st.metric(label="AZ (Accelerometer Z)", value=f"{az:.6f}")

        # Display the gyroscope data
        st.subheader("Gyroscope Data:")
        st.metric(label="GX (Gyroscope X)", value=f"{gx:.6f}")
        st.metric(label="GY (Gyroscope Y)", value=f"{gy:.6f}")
        st.metric(label="GZ (Gyroscope Z)", value=f"{gz:.6f}")

    return ax, ay, az, gx, gy, gz


# Streamlit app layout
st.title('Fall Detection Prediction')

# Create two columns
col1, col2 = st.columns(2)

# Column 1 for Refresh Data
with col1:
    if st.button('Refresh Data'):
        accelerometer_data, gyroscope_data = get_sensor_data()
        display_data(accelerometer_data, gyroscope_data, col1)

# Column 2 for Predict
with col2:
    if st.button('Predict'):
        # Fetch the latest data
        accelerometer_data, gyroscope_data = get_sensor_data()
        ax, ay, az, gx, gy, gz = display_data_(accelerometer_data, gyroscope_data)

        # Proceed with prediction
        prediction = predict_outcome([ax, ay, az, gx, gy, gz])
        # Using an expander to simulate a modal
        with st.expander("See Prediction Result", expanded=True):
            if prediction[0]:
                # Fall Detected
                st.markdown(f"<div style='background-color:lightcoral; padding: 10px; border-radius: 5px;'> <h2 style='color: white;'>Fall Detected, Dialing 911....</h2></div>", unsafe_allow_html=True)
            else:
                # No Fall Detected
                st.markdown(f"<div style='background-color:lightgreen; padding: 10px; border-radius: 5px;'><h2 style='color: white;'>No Fall Detected</h2></div>", unsafe_allow_html=True)
