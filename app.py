import numpy as np
import joblib
import streamlit as st

st.title('Salary prediction app')
st.divider()
st.write('With this app you can get estimation for the salaries of the company employees.')

# Added labels to st.number_input
years = st.number_input('Years of Experience', value=1, step=1, min_value=0)
jobrate = st.number_input('Job Rate (1-5)', value=3.5, step=0.5, min_value=0.0, max_value=5.0) # Added max_value based on previous analysis

X = [years, jobrate]
model = joblib.load('Random forest regressor model.pkl')
st.divider()

predict = st.button('Press the button for salary prediction')

st.divider()

if predict:
    st.balloons()
    X1 = np.array([X])
    prediction = model.predict(X1)[0]
    st.write(f'Salary prediction is: ${prediction:,.2f}')
else:
    st.info('Please press the button in order for the app to make predictions.')