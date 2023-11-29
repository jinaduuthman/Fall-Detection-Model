import streamlit as st
import pandas as pd
import joblib
from firebase import firebase
import streamlit as st
import time

# Load the trained model (ensure it's in the same directory or provide the full path)
rf_model = joblib.load('random_forest_model.pkl')
# rf_model = joblib.load('best_gb_model.pkl')


#Firebase settings config
firebase_url = 'https://falldetection-6c89a-default-rtdb.firebaseio.com/'
firebase_db = firebase.FirebaseApplication(firebase_url, None)


# Initialize session state variables for data storage
if 'accelerometer_data' not in st.session_state:
    st.session_state['accelerometer_data'] = None

if 'gyroscope_data' not in st.session_state:
    st.session_state['gyroscope_data'] = None

if 'data_refreshed' not in st.session_state:
    st.session_state['data_refreshed'] = False


def predict_outcome(features):
    # Convert input data into a DataFrame
    # input_data = pd.DataFrame([features], columns=['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'])
    input_data = pd.DataFrame([features], columns=['AX', 'AY'])
    
    # Predict
    prediction = rf_model.predict(input_data)
    return prediction

def get_sensor_data():
    # 从 Firebase 获取加速度计和陀螺仪数据
    accelerometer_data = firebase_db.get('/Accelerometer', None)
    gyroscope_data = firebase_db.get('/Gyroscope', None)
    
    return accelerometer_data, gyroscope_data

def get_gps_data():
    # Fetch GPS data from Firebase
    gps_data = firebase_db.get('/GPS', None)
    longitude = gps_data.get('longitude', 'Not available')
    latitude = gps_data.get('latitude', 'Not available')
    
    return longitude, latitude


#THis version of display_data- is only used for the predict button.
def display_data_(accelerometer_data, gyroscope_data):
    # Extract the accelerometer and gyroscope data
    ax = accelerometer_data.get('x', 0) if accelerometer_data else 0
    ay = accelerometer_data.get('y', 0) if accelerometer_data else 0
    az = accelerometer_data.get('z', 0) if accelerometer_data else 0

    gx = gyroscope_data.get('x', 0) if gyroscope_data else 0
    gy = gyroscope_data.get('y', 0) if gyroscope_data else 0
    gz = gyroscope_data.get('z', 0) if gyroscope_data else 0

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

# Fetch and display data, and make predictions automatically
try:
    # Fetch new data
    accelerometer_data, gyroscope_data = get_sensor_data()
    if accelerometer_data and gyroscope_data:
        col1, col2 = st.columns(2)
        with col1:
            display_data(accelerometer_data, gyroscope_data, col1)

        with col2:
            # Extract data for prediction
            ax, ay, az, gx, gy, gz = display_data_(accelerometer_data, gyroscope_data)

            # Make prediction
            # prediction = predict_outcome([ax, ay, az, gx, gy, gz])
            prediction = predict_outcome([ax, ay])


            # Display prediction result
            with st.expander("See Prediction Result", expanded=True):
                if prediction[0]:
                    st.markdown(f"<div style='background-color:lightcoral; padding: 10px; border-radius: 5px;'> <h2 style='color: white;'>Fall Detected, Dialing 911....</h2></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:lightgreen; padding: 10px; border-radius: 5px;'><h2 style='color: white;'>No Fall Detected</h2></div>", unsafe_allow_html=True)
            
            longitude, latitude = get_gps_data()  # Fetch GPS data
            # Display the location on a map
            map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_data)
            
    else:
        st.warning("No data available. Please check the data source.")
    
    # Rerun the app every 10 seconds
    time.sleep(10)
    st.experimental_rerun()

except Exception as e:
    st.error(f"An error occurred: {e}")