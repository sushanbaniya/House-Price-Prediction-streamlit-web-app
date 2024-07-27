import streamlit as st
import joblib 
import numpy as np
from streamlit_extras.let_it_rain import rain



model = joblib.load('model.pkl')

st.title('App for Predicting House Price using Data Science and Machine Learning')

st.divider()

st.write(':blue[ENTER INPUTS AND CLICK PREDICT BUTTON]')

st.divider()

bedrooms = st.number_input('Enter number of bedrooms', min_value=0, value=0)
bathrooms = st.number_input('Enter number of bathrooms', min_value=0, value=0)
livingarea = st.number_input('Living Area', min_value=0, value=2000)
condition = st.number_input('Condition', min_value=0, value=4)
numberofschools = st.number_input('Number of Schools Nearby', min_value=0, value=0)

st.divider()

X = [[bedrooms,bathrooms,livingarea,condition,numberofschools]]

predictbutton = st.button('PREDICT THE HOUSE PRICE')

if predictbutton:
    rain(emoji='🙂') 
    X_array = np.array(X)
    prediction = model.predict(X_array)[0]
    st.write('The Price of the House is: ', prediction)

    ## test code below
    # prediction = model.predict(X)
    # st.write('price is: ', prediction)
    ##

    # print(type(X))
    # print(type(X_array))
    # print(type(model.predict(X_array)))
    # print(model.predict(X_array))



# else:
#     st.write('Click Predict Button after entering Input')
