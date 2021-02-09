import streamlit as st
import pickle as pkl


st.title('Operating System Predictor')
#st.image()

st.slider('Age', min_value = 0, max_value = 80,
         value = 30,  key = 'age')

st.selectbox('Select Gender', 
            ['Male', 'Female', 'Gender Non-Conforming'],
            key = 'gender')

st.selectbox('Select Region', ['North America',
        'South America', 'Asia', 'Europe',
        'Australia', 'Middle East', 
        'Commonwealth of Independant States',
        'Africa', 'Baltics'], key = 'region')

st.selectbox('Select Education Level', 
            ['Masters/PHD',
             'Bachelors ', 'Associates',
            'Professional Degree', 'Some University',
            'Secondary or Primary School', 
            'None'], key = 'edlevel')

st.selectbox('Select Occupation/Developer Type', 
            ['Not a Developer', 'Sometimes Code at Work',
             'Retired Developer', 'Back-End Developer', 
             'Full-Stack Developer', 'Front-End Developer', 
             'Desktop', 'Mobile', 'DevOps', 'Database Administrator', 
             'Designer', 'System Administrator', 
             'Student'], key = 'jobtype')

st.selectbox('Select Undergrad Major', ['Computer Science/Engineering',
        'Engineering', 'Information Technology/Systems',
        'Health/Natural Sciences', 'Human/Social Sciences', 
        'Math/Statistics', 'Web Development',
        'Arts', 'None'], key = 'major')

st.multiselect('Select Coding Languages', 
                ['JavaScript', 'Python', 'SQL', 'HTML/CSS', 'Java'],
                key = 'language')

with open("../Models/xgb_balanced.pkl", "rb") as f:
    xgb_model = pkl.load(f)

if st.button('Get Prediction', 'predict'):

    OpSys = xgb_model.feature_names
    st.write('The Prediction is: ', OpSys)

