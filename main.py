import streamlit as st
from function import load_data
import pandas as pd
import numpy as np
from Tabs import home, predict, visualise

#membuat tab
Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualisation": visualise
}

#sidebar
st.sidebar.title("Navbar")

page = st.sidebar.radio("Pages",list(Tabs.keys()))

#load dataset
df, x, y = load_data()

#----------------------
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()