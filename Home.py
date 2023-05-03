import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

print("*****************")

st.title("CloudCast: Weather Trend Prediction System")
st.markdown("##### A Big Data Approach Towards Predicting Weather Trend")

team = pd.DataFrame({"Name":['Samarth Tumdi', 'Dhruvi Shah', 'Umang Lodaya'], "SAP ID":['60009200015', '60009200025', '60009200032']})
st.write("")
st.write("Our Team:")
st.write(team)
# st.sidebar.title("Navigation")

print("**DONE**\n\n")

