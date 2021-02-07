import streamlit as st


st.title('Operating System Predictor')
#st.image()

st.select('Select Gender', 
            ['Male', 'Female', 'Gender Non-Conforming'],
            'gender')

st.select('Select Region', ['North America',
        'South America', 'Asia', 'Europe',
        'Australia', 'Middle East', 
        'Commonwealth of Independant States',
        'Africa', 'Baltics'], 'region')

st.multiselect('Select Coding Languages', 
                ['JavaScript', 'Python', 'SQL', 'HTML/CSS', 'Java'],
                key = 'language')

if st.button('Get Prediction', 'predict'):

    OpSys = 'Windows'
    st.write('The Prediction is: ', OpSys)

