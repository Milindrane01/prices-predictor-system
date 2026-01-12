import streamlit as st
import json
import requests
import pandas as pd

st.title("House Price Prediction System")

st.markdown("""
This app sends a prediction request to your running MLflow deployment.
Make sure you have run `python run_deployment.py` before using this.
""")

# Input URL
url = st.text_input("Prediction Service URL", "http://127.0.0.1:8000/invocations")

st.header("Housing Features")

# Create some key inputs for interactivity
col1, col2 = st.columns(2)
with col1:
    lot_area = st.number_input("Lot Area", value=9600)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    year_built = st.number_input("Year Built", value=1961)
with col2:
    gr_liv_area = st.number_input("Gr Liv Area (sq ft)", value=1710)
    garage_cars = st.number_input("Garage Cars", value=2)
    total_bsmt_sf = st.number_input("Total Basement SF", value=850)

# Default data payload (based on sample_predict.py structure)
input_data = {
    "dataframe_records": [
        {
            "Order": 1,
            "PID": 5286,
            "MS SubClass": 20,
            "Lot Frontage": 80.0,
            "Lot Area": lot_area,
            "Overall Qual": overall_qual,
            "Overall Cond": 7,
            "Year Built": year_built,
            "Year Remod/Add": 1961,
            "Mas Vnr Area": 0.0,
            "BsmtFin SF 1": 700.0,
            "BsmtFin SF 2": 0.0,
            "Bsmt Unf SF": 150.0,
            "Total Bsmt SF": total_bsmt_sf,
            "1st Flr SF": 856,
            "2nd Flr SF": 854,
            "Low Qual Fin SF": 0,
            "Gr Liv Area": gr_liv_area,
            "Bsmt Full Bath": 1,
            "Bsmt Half Bath": 0,
            "Full Bath": 1,
            "Half Bath": 0,
            "Bedroom AbvGr": 3,
            "Kitchen AbvGr": 1,
            "TotRms AbvGrd": 7,
            "Fireplaces": 2,
            "Garage Yr Blt": 1961,
            "Garage Cars": garage_cars,
            "Garage Area": 500.0,
            "Wood Deck SF": 210.0,
            "Open Porch SF": 0,
            "Enclosed Porch": 0,
            "3Ssn Porch": 0,
            "Screen Porch": 0,
            "Pool Area": 0,
            "Misc Val": 0,
            "Mo Sold": 5,
            "Yr Sold": 2010,
        }
    ]
}

# Advanced view: Show raw JSON being sent
if st.checkbox("Show Raw JSON Payload"):
    st.json(input_data)

if st.button("Predict"):
    try:
        # Convert to JSON
        json_data = json.dumps(input_data)
        headers = {"Content-Type": "application/json"}
        
        # Send POST request
        response = requests.post(url, headers=headers, data=json_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful!")
            
            # Display result nicely
            prediction_value = result.get('predictions', [None])[0]
            if prediction_value is not None:
                st.metric(label="Predicted House Price", value=f"${prediction_value:,.2f}")
            else:
                st.write(result)
        else:
            st.error(f"Error: {response.status_code}")
            st.text(response.text)
            
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
