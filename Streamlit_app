import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('car_price_model.pkl')
encoders = joblib.load('label_encoders.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('🚗 Car Price Prediction')

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox('Make', encoders['Make'].classes_)
    model_name = st.selectbox('Model', encoders['Model'].classes_)
    year = st.number_input('Year', min_value=1990, max_value=2026, value=2020)
    mileage = st.number_input('Mileage (km)', min_value=0, value=50000)
    fuel = st.selectbox('Fuel', encoders['Fuel'].classes_)

with col2:
    location = st.selectbox('Location', encoders['Location'].classes_)
    trim = st.selectbox('Trim', encoders['Trim'].classes_)
    seats = st.number_input('Seats', min_value=2, max_value=12, value=5)

if st.button('Predict Price'):
    age = 2026 - year
    input_dict = {
        'Year': year,
        'Mileage': mileage,
        'Seats': seats,
        'Age': age,
        'Fuel': encoders['Fuel'].transform([fuel])[0],
        'Location': encoders['Location'].transform([location])[0],
        'Make': encoders['Make'].transform([make])[0],
        'Model': encoders['Model'].transform([model_name])[0],
        'Trim': encoders['Trim'].transform([trim])[0],
    }
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    pred_log = model.predict(input_df)
    pred_price = np.expm1(pred_log)[0]

    st.success(f'### Estimated Price: PKR {pred_price:,.0f}')
    st.info(f'Price range: PKR {pred_price*0.9:,.0f} — PKR {pred_price*1.1:,.0f}')

with open('car_price_app/requirements.txt', 'w') as f:
    f.write(requirements)

print("All files:")
print(os.listdir('car_price_app'))
