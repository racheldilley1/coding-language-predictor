import streamlit as st


st.title('Operating System Predictor')
#st.image()

if st.button('Get Prediction', 'predict'):

    OpSys = 'Windows'
    st.write('The Prediction is: ', OpSys)

