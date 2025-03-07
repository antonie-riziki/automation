import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
import plotly.express as px
import os
import csv
import sys


registration_page = st.Page("./pages/registration.py", title="signup", icon=":material/app_registration:")
home_page = st.Page("./pages/homepage.py", title="Home", icon=":material/home:")
automation_page = st.Page("./pages/automation.py", title="automation", icon=":material/automation:")




pg = st.navigation([registration_page, home_page, automation_page])


st.set_page_config(
    page_title="EFH Innovation",
    page_icon="🌱🪴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.echominds.africa',
        'Report a bug': "https://www.echominds.africa",
        'About': "# We are a leading insights and predicting big data application, Try *EFH Innovation* and experience reality!"
    }
)

pg.run()