
import os
import streamlit as st 

st.title("Settings")

# Store the initial value of widgets in session state
if "model" not in st.session_state:
    st.session_state.model = "svm.sav"
    

models = os.listdir('./models')

st.session_state.model = st.selectbox(
    'Chose model for face recognition',
    (models),
    index=models.index(st.session_state.model)
    )



