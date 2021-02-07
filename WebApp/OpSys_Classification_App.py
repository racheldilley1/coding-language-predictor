import streamlit as st


st.title('Operating System Predictor')
#st.image()

st.multiselect('Select Coding Languages', 
                ['JavaScript', 'Python', 'SQL', 'HTML/CSS', 'Java'],
                key = 'language')

if st.button('Get Prediction', 'predict'):

    OpSys = 'Windows'
    st.write('The Prediction is: ', OpSys)

