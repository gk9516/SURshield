# import streamlit as st
# import pandas as pd
# import joblib
# import time
# from sklearn.preprocessing import LabelEncoder

# # Load the model and encoders
# model = joblib.load('logistic_regression_model.pkl')

# # Load the CSV file and LabelEncoders for transformation
# data = pd.read_csv('expanded_synthetic_health_data (1).csv')
# X = data.drop(columns='Severity')  # Use this to check for missing features

# label_encoders = {}
# for column in data.select_dtypes(include=['object']).columns:
#     label_encoders[column] = LabelEncoder()
#     label_encoders[column].fit(data[column])

# # Conversion dictionary for manual conversions
# conversion_dict = {
#     "Gender": {"Female": 0, "Male": 1, "Unknown": 2},
#     "Smoking": {"No": 0, "Yes": 1},
# }

# # Function to convert user inputs to the model-compatible format without showing warnings
# def convert_input(column, value):
#     if column in conversion_dict:
#         return conversion_dict[column].get(value, value)

#     # Handle unseen labels without showing warnings
#     if column in label_encoders:
#         try:
#             return label_encoders[column].transform([value])[0]
#         except ValueError:
#             # Silently handle unseen label by using the first available class (default label)
#             return label_encoders[column].transform([label_encoders[column].classes_[0]])[0]
#     return value

# # Ensure feature names are consistent
# def adjust_feature_names(user_data):
#     feature_mapping = {
#         'Pain Scale (0-10)': 'Pain Scale',
#         'Stress Levels (0-10)': 'Stress Levels'
#     }
#     adjusted_data = {}
#     for feature, value in user_data.items():
#         # Map the feature name if needed
#         adjusted_feature = feature_mapping.get(feature, feature)
#         adjusted_data[adjusted_feature] = value
#     return adjusted_data

# # Function to show loading spinner and predict severity
# def predict_severity(user_data):
#     # Adjust the feature names to match the model's features
#     adjusted_data = adjust_feature_names(user_data)
    
#     # Simulate a loading process (displayed after submission)
#     with st.spinner('Predicting severity...'):
#         time.sleep(2)  # Simulate delay for prediction processing

#         # Convert inputs
#         model_input = {}
#         for column, value in adjusted_data.items():
#             model_input[column] = [convert_input(column, value)]
#         user_df = pd.DataFrame(model_input)

#         # Check for missing columns and add them with default values
#         for col in X.columns:
#             if col not in user_df.columns:
#                 user_df[col] = 0  # Default value for missing features

#         # Make prediction
#         severity_prediction = model.predict(user_df)
#         severity_class = label_encoders['Severity'].inverse_transform(severity_prediction)

#         return severity_class[0]

# # Survey Form
# def show_survey():
#     st.title('MEDICAL SURVEY PAGE')
#     st.write('Please fill out the following questionnaire to asses severity:')

#     user_data = {}
#     questions = {
#         'Gender': ['Male', 'Female', 'Unknown'],
#         'General Symptoms': ['Fatigue', 'Cough', 'Shortness of Breath'],
#         'Pain Scale (0-10)': 'number',
#         'Symptom Duration': ['Less than 2 days', '2-5 days', 'More than 5 days'],
#         'Onset': ['Sudden', 'Gradual'],
#         'Chronic Conditions': ['Hypertension', 'Diabetes', 'Asthma', 'None'],
#         'Allergies': ['Yes', 'No'],
#         'Medications': ['Yes', 'No'],
#         'Travel History': ['Yes', 'No'],
#         'Contact with Sick Individuals': ['Yes', 'No'],
#         'Smoking': ['Yes', 'No'],
#         'Alcohol Consumption': ['Yes', 'No', 'Occasionally'],
#         'Physical Activity': ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never'],
#         'Stress Levels (0-10)': 'number',
#         'Sleep Quality': ['Very Good', 'Good', 'Average', 'Poor', 'Very Poor']
#     }

#     # Age as a number input instead of a slider
#     user_data['Age'] = st.number_input('Age:', min_value=1, max_value=100, step=1)

#     for question, input_type in questions.items():
#         if input_type == 'number':
#             user_data[question] = st.slider(f'{question}', min_value=0, max_value=10, step=1)
#         else:
#             user_data[question] = st.selectbox(f'{question}', input_type)

#     submit_button = st.button('Submit', key='submit_button', help="Submit the survey", 
#                               use_container_width=True)

#     if submit_button:
#         return user_data
#     return None

# # Main Streamlit Application
# def main():
#     st.set_page_config(page_title="survey page", layout="centered", initial_sidebar_state="collapsed")
    
#     # Set background and text color
#     st.markdown("""
#         <style>
#         .main {
#             background-color: #c2e3fe;
#             color: #333;`
#         }
#         h1 {
#             color: #5a5a87;
            
#         }
#         .stButton>button {
#             background-color: #c2e3fe;
#             color: white;
#             width: 100%;
#             height: 60px;
#             font-size: 18px;
#             border-radius: 10px;
#         }
#         .stSlider>div>div>div>div>div>div {
#             color: #5a5a87;
#         }
#         .stSelectbox>div>div>div>div {

#             color: #333;
#         }
#         </style>
#         """, unsafe_allow_html=True)

#     if 'user_data' not in st.session_state:
#         st.session_state['user_data'] = None

#     if not st.session_state['user_data']:
#         user_data = show_survey()
#         if user_data:
#             st.session_state['user_data'] = user_data
#             st.experimental_rerun()
            
#     else:
#         # Display result page
#         st.title("Severity Prediction Result")
#         st.write("### your condition based on your response seems to be:")

#         # Box for severity result
#         severity = predict_severity(st.session_state['user_data'])
#         st.markdown(f"<div style='background-color:#b3d9ff;padding:20px;text-align:center;border-radius:10px;'><h1>{severity}</h1></div>", unsafe_allow_html=True)

#         st.write("### Survey Responses given by you")
#         # Display user data in a table format
#         table_data = pd.DataFrame(list(st.session_state['user_data'].items()), columns=["Question", "Responses"])
        
#         # Apply custom styles to make the table bold
#         styled_table = table_data.style.set_properties(**{
#             'text-align': 'left',
#             'font-weight': 'bold',
#             'background-color': '#f0f8ff',
#             'border': '2px solid black'
#         })
        
#         st.write(styled_table.to_html(), unsafe_allow_html=True)

# if __name__ == '__main__':
#     main()

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model('severity_analysis_model.h5')

# Load encoders and scaler (assuming these objects were pickled)
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define mappings
gender_mapping = {'Male': 1, 'Female': 0}
symptom_mapping = {
    "Abdominal pain": 1.0, "Chest pain": 2.0, "Constipation": 3.0, "Cough": 4.0, "Diarrhea": 5.0,
    "Difficulty swallowing": 6.0, "Dizziness": 7.0, "Eye discomfort and redness": 8.0,
    "Foot pain or ankle pain": 9.0, "Foot swelling or leg swelling": 10.0, "Headaches": 11.0,
    "Heart palpitations": 12.0, "Hip pain": 13.0, "Knee pain": 14.0, "Low back pain": 15.0,
    "Nasal congestion": 16.0, "Nausea or vomiting": 17.0, "Neck pain": 18.0, "Numbness or tingling in hands": 19.0,
    "Shortness of breath": 20.0, "Shoulder pain": 21.0, "Sore throat": 22.0, "Urinary problems": 23.0,
    "Wheezing": 24.0, "Ear ache": 25.0, "Fever": 26.0, "Joint pain or muscle pain": 27.0, "Skin rashes": 28.0
}
onset_mapping = {'Sudden': 1, 'Gradual': 0}
symptom_duration_mapping = {'Less than 2 days': 0, '2-5 days': 1, 'More than 5 days': 2}
chronic_mapping = {
    "Diabetes": 1.0, "Hypertension": 2.0, "Asthma": 3.0, "Arthritis": 4.0, "Obesity": 5.0,
    "Cholesterol": 6.0, "Depression": 7.0, "Cirrhosis": 8.0, "No chronic conditions": 9.0
}
alcohol_mapping = {'No': 0, 'Occasionally': 1, 'Regularly': 2}
physical_mapping = {'No': 0, 'Light': 1, 'Moderate': 2, 'Intense': 3}
sleep_mapping = {'Excellent': 3.0, 'Good': 2.0, 'Fair': 1.0, 'Poor': 0.0}

# Function to show prescription based on severity
def show_prescription(severity):
    st.markdown("<div style='background-color:#b3d9ff;padding:20px;text-align:center;border-radius:10px;'>", unsafe_allow_html=True)
    if severity == 'Mild':
        st.write("### Prescription:")
        st.write("- Rest without any disturbance.")
        st.write("- Drink a lot of water.")
        st.write("- Don't stress too much.")
        st.write("- Try to avoid processed food.")
    elif severity == 'Moderate':
        st.write("### Prescription:")
        st.write("- Rest without any disturbance.")
        st.write("- Drink a lot of water.")
        st.write("- Don't stress too much.")
        st.write("- Avoid alcohol and cigarettes if possible.")
        st.write("- Don't sit for too long; touch some grass.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Book an Appointment"):
                st.session_state['action'] = 'appointment'
                st.rerun()
        with col2:
            if st.button("Order Medicines"):
                st.session_state['action'] = 'medicines'
                st.rerun()
    elif severity == 'Severe':
        st.write("### Warning:")
        st.write("- Do not do anything to counter the effect.")
        st.write("### Prescription:")
        st.write("- Do not take any medication on your own.")
        st.write("- Rest without any disturbance.")
        st.write("- Drink a lot of water.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Special Appointment"):
                st.session_state['action'] = 'special_appointment'
                st.rerun()
        with col2:
            if st.button("Priority Medicines"):
                st.session_state['action'] = 'priority_medicines'
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Function to show the survey inputs and collect user data
def show_survey():
    user_data = {}

    user_data['Gender'] = st.selectbox('Gender', options=['Select Gender', 'Male', 'Female'], index=0, key='gender')  
    user_data['Age'] = st.number_input('Age', min_value=0, max_value=120, step=1, key='age')
    user_data['General Symptoms'] = st.selectbox('General Symptoms', options=['Select Symptom'] + list(symptom_mapping.keys()), index=0, key='symptoms')
    user_data['Pain Scale'] = st.slider('Pain Scale (0-10)', 0, 10, key='pain_scale', value=0)
    user_data['Symptom Duration'] = st.selectbox('Symptom Duration', options=['Select Duration'] + list(symptom_duration_mapping.keys()), index=0, key='duration')
    user_data['Onset'] = st.selectbox('Onset', options=['Select Onset', 'Sudden', 'Gradual'], index=0, key='onset')
    user_data['Chronic Conditions'] = st.selectbox('Chronic Conditions', options=['Select Condition'] + list(chronic_mapping.keys()), index=0, key='chronic')
    user_data['Allergies'] = st.radio('Do you have allergies?', options=['Please select', 'Yes', 'No'], index=0, key='allergies')
    user_data['Medications'] = st.radio('Do you take medications?', options=['Please select', 'Yes', 'No'], index=0, key='medications')
    user_data['Travel History'] = st.radio('Recent travel history?', options=['Please select', 'Yes', 'No'], index=0, key='travel')
    user_data['Contact with Sick Individuals'] = st.radio('Contact with sick individuals?', options=['Please select', 'Yes', 'No'], index=0, key='contact')
    user_data['Smoking'] = st.radio('Do you smoke?', options=['Please select', 'Yes', 'No'], index=0, key='smoking')
    user_data['Alcohol Consumption'] = st.selectbox('Alcohol Consumption', options=['Select Alcohol Consumption'] + ['No', 'Occasionally', 'Regularly'], index=0, key='alcohol')
    user_data['Physical Activity'] = st.selectbox('Physical Activity', options=['Select Activity'] + ['No', 'Light', 'Moderate', 'Intense'], index=0, key='activity')
    user_data['Stress Levels'] = st.slider('Stress Levels (0-10)', 0, 10, key='stress', value=0)
    user_data['Sleep Quality'] = st.selectbox('Sleep Quality', options=['Select Sleep Quality'] + ['Excellent', 'Good', 'Fair', 'Poor'], index=0, key='sleep')

    if st.button("Submit"):
        # Ensure all fields are filled before returning data
        if all(value != 'Select Gender' and value != 'Select Symptom' and value != 'Select Duration' and 
                value != 'Select Onset' and value != 'Select Condition' and value != 'Please select' 
                for value in user_data.values()):
            return user_data
        else:
            st.warning("Please fill all fields before submitting.")
    return None

# Function to predict severity based on user data
def predict_severity(user_data):
    # Process the user inputs and map to appropriate values
    user_data['Gender'] = gender_mapping[user_data['Gender']]
    user_data['General Symptoms'] = symptom_mapping[user_data['General Symptoms']]
    user_data['Pain Scale'] = user_data['Pain Scale'] / 10.0
    user_data['Onset'] = onset_mapping[user_data['Onset']]
    user_data['Symptom Duration'] = symptom_duration_mapping[user_data['Symptom Duration']]
    user_data['Chronic Conditions'] = chronic_mapping[user_data['Chronic Conditions']]
    user_data['Allergies'] = 1 if user_data['Allergies'] == 'Yes' else 0
    user_data['Medications'] = 1 if user_data['Medications'] == 'Yes' else 0
    user_data['Travel History'] = 1 if user_data['Travel History'] == 'Yes' else 0
    user_data['Contact with Sick Individuals'] = 1 if user_data['Contact with Sick Individuals'] == 'Yes' else 0
    user_data['Smoking'] = 1 if user_data['Smoking'] == 'Yes' else 0
    user_data['Alcohol Consumption'] = alcohol_mapping[user_data['Alcohol Consumption']]
    user_data['Physical Activity'] = physical_mapping[user_data['Physical Activity']]
    user_data['Stress Levels'] = user_data['Stress Levels'] / 10.0
    user_data['Sleep Quality'] = sleep_mapping[user_data['Sleep Quality']]

    # Convert the user input into a DataFrame
    user_df = pd.DataFrame([user_data])

    # Apply scaler
    numeric_columns = ['Age', 'Symptom Duration', 'Chronic Conditions', 'Alcohol Consumption', 'Physical Activity', 'Sleep Quality']
    user_df[numeric_columns] = scaler.transform(user_df[numeric_columns])

    # Make prediction
    prediction = model.predict(user_df)
    severity_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return severity_class[0]

# Function to display appointment confirmation
def confirm_appointment(severity, user_data):
    st.title("Confirm Appointment")
    st.write(f"### Severity: {severity}")
    st.write("### Your Responses:")
    table_data = pd.DataFrame(list(user_data.items()), columns=["Question", "Responses"])
    st.write(table_data)
    if st.button("Confirm Appointment"):
        st.success("Your appointment has been confirmed!")
        # Redirect to a blank page
        st.rerun()

# Function to display medicine invoice
def medicine_invoice():
    st.title("Medicine Invoice")
    medicines = {
        'Cough Syrup': 100,
        'Pain Reliever': 50,
        'Cold Medicine': 75,
        'Allergy Medication': 120
    }
    st.write("### Medicines based on your symptoms:")
    invoice_data = pd.DataFrame(list(medicines.items()), columns=["Medicine", "Cost"])
    st.write(invoice_data)
    if st.button("Confirm Bill"):
        st.success("Your bill has been confirmed!")
        # Redirect to a blank page
        st.rerun()

# Streamlit Page Configuration
st.set_page_config(page_title="Survey Page", layout="centered", initial_sidebar_state="collapsed")

# Set background and text color
st.markdown("""<style>
    .main {
        background-color: #c2e3fe;
        color: #333;
    }
    h1 {
        color: #5a5a87;
    }
    .stButton>button {
        background-color: #c2e3fe;
        color: white;
        width: 100%;
        height: 60px;
        font-size: 18px;
        border-radius: 10px;
    }
    .stSlider>div>div>div>div>div>div {
        color: #5a5a87;
    }
    .stSelectbox>div>div>div>div {
        color: #333;
    }
</style>""", unsafe_allow_html=True)

# Initialize session state variables
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = None
    st.session_state['action'] = None

# Main logic for user interaction
if st.session_state['action']:
    if st.session_state['action'] == 'appointment':
        confirm_appointment(st.session_state['severity'], st.session_state['user_data'])
    elif st.session_state['action'] == 'medicines':
        medicine_invoice()
    elif st.session_state['action'] == 'special_appointment':
        confirm_appointment('Severe', st.session_state['user_data'])
    elif st.session_state['action'] == 'priority_medicines':
        medicine_invoice()

if not st.session_state['user_data']:
    user_data = show_survey()
    if user_data:
        st.session_state['user_data'] = user_data
        st.session_state['severity'] = predict_severity(user_data)
        st.rerun()
else:
    # Display result page
    st.title("Severity Prediction Result")
    st.write("### Your condition based on your response seems to be:")
    severity = st.session_state['severity']
    st.markdown(f"<div style='background-color:#b3d9ff;padding:20px;text-align:center;border-radius:10px;'>"
                f"<h2 style='color:#5a5a87;'>{severity}</h2></div>", unsafe_allow_html=True)
    show_prescription(severity)
