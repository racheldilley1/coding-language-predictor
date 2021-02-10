import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np


model = pkl.load(open("../Models/xgb_balanced.pkl", "rb"))   

st.title('Operating System Predictor')
#st.image()

age = st.slider('Age', min_value = 0, max_value = 80,
         value = 30)

gender = st.selectbox('Select Gender', 
                    ['Choose an option', 'Male', 'Female', 'Gender Non-Conforming']
                    )

region = st.selectbox('Select Region', ['Choose an option', 'North America',
                        'South America', 'Asia', 'Europe',
                        'Australia', 'Middle East', 
                        'Commonwealth of Independant States',
                        'Africa', 'Baltics', 'Other'])

edlevel = st.selectbox('Select Education Level', 
                        ['Choose an option', 'Masters/PHD',
                        'Bachelors', 'Associates',
                        'Professional Degree', 'Some University',
                        'Secondary or Primary School', 
                        'No Education'])

major = st.multiselect('Select Undergrad Major', ['Computer Science',
                    'Engineering', 'Information Technology/Systems',
                    'Health/Natural Sciences', 'Human/Social Sciences', 
                    'Math/Statistics', 'Web Development',
                    'Arts', 'None'])

jobtype = st.multiselect('Select Occupation/Developer Type', 
                        ['Not a Developer', 'Sometimes Code at Work',
                        'Retired Developer', 'Back-End Developer', 
                        'Full-Stack Developer', 'Front-End Developer', 
                        'Desktop Developer', 'Mobile Developer', 'DevOps Specialist', 'Database Administrator', 
                        'Designer', 'System Administrator', 
                        'Student'])

age1stcode = st.slider('At what age did you start coding?', min_value = 10, max_value = 60,
         value = 20)

yearscode = st.slider('How many years have you coded?', min_value = 0, max_value = 50,
         value = 10)

yearscodepro = st.slider('How many years have you coded professionaly?', min_value = 0, max_value = 50,
         value = 20)

language = st.multiselect('What coding languages are you familiar with?', 
                        ['JavaScript', 'Python', 'SQL', 'HTML/CSS', 'Java'])

databases = st.multiselect('What database environments are you familiar with?', 
                        ['Cassandra', 'Couchbase', 'DynamoDB', 'Elasticsearch', 'Firebase', 'IBM DB2', 'MariaDB', 'Microsoft SQL Server',
                        'MongoDB', 'MySQL', 'Oracle', 'PostgreSQL', 'Redis', 'SQLite'])

# databases = st.slider('How many databases have you used?', min_value = 0, max_value = 15,
#          value = 0)

        #                 '', 'Associates',
        #                 'Secondary or Primary School', 
        #                 ''])
        #             'Arts', 
        #             'Male' 
        #                 'Africa', 

model_params = ['databases', 'age1stcode', 'yearscodepro', 'age', 'yearscode',
                'Bachelors', 'Masters/PHD',  'Professional Degree',
                'Some University', 'Current Student', 'Female',
                 'Computer Science',
                'Engineering', 'Health/Natural Sciences',
                 'Information Technology/Systems',
                'Math/Statistics', 'None',
                'Web Development', 'Asia', 'Australia',
                 'Europe', 'Middle East',
                'North America', 'South America', 'Other', 
                'Back-End Developer', 'Front-End Developer', 'Desktop Developer',
                'Mobile Developer', 'DevOps Specialist', 'Database Administrator', 
                'System Administrator', 'Student', 
                'Python', 'SQL', 'Java', 'HTML/CSS']

input_variables = pd.DataFrame(columns = model_params)

if st.button('Get Prediction'):
    input_variables.loc[0, ['databases']] = len(databases)
    input_variables.loc[0,['age1stcode']] = age1stcode
    input_variables.loc[0,['yearscodepro']] = yearscodepro
    input_variables.loc[0,['age']] = age
    input_variables.loc[0,['yearscode']] = yearscode

    input_vals = [gender] + [region] + [edlevel] + major + jobtype + language

    for x in input_vals:
        for y in list(input_variables.columns):
            if x in y:
                input_variables.loc[0, [y]] = 1

    input_variables = input_variables.fillna(0)

    input_variables.columns = ['database_count', 'Age1stCode', 'YearsCodePro', 'Age', 'YearsCode', 'EdLevel_BA/BS', 
                                        'EdLevel_MA/PhD', 'EdLevel_Prof', 'EdLevel_Some Univ', 'EdLevel_Student', 'Gender_Woman', 
                                        'UndergradMajor_Comp Sci/Eng', 'UndergradMajor_Eng', 'UndergradMajor_Health/Nat Sci', 'UndergradMajor_Info Tech/Sys', 
                                        'UndergradMajor_Math/Stats', 'UndergradMajor_None', 'UndergradMajor_Web Dev', 'Region_Asia', 'Region_Australia', 
                                        'Region_Europe', 'Region_M East', 'Region_N America', 'Region_S America', 'Region_other', 'back-end_Yes', 'front-end_Yes', 
                                        'desktop_Yes', 'mobile_Yes', 'DevOps_Yes', 'Database admin_Yes', 'System admin_Yes', 'Student_Yes', 'Python_Yes', 
                                        'SQL_Yes', 'Java_Yes', 'HTML/CSS_Yes']
    
    #st.dataframe(input_variables)
    OpSys = model.predict(input_variables.iloc[0:1,:])
    probs = model.predict_proba(input_variables.iloc[0:1,:] >= 0.4)
    #st.write('Prediction: ', OpSys[0])
    linux = np.round(probs[0,0]*100)
    mac = np.round(probs[0,1]*100)
    windows = np.round(probs[0,2]*100)
    st.write(str(linux) + "% probability Linux-based")
    st.write(str(mac) + "% probability MacOS")
    st.write(str(windows) + "% probability Windows")



