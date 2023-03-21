import streamlit as st
import pandas as pd
import os
import pickle
import sklearn 
import scipy
import numpy as np



# first line after the importation section
st.set_page_config(page_title="Sales predictor app", layout="centered")

@st.cache_resource()  # stop the hot-reload to the function just bellow

def prediction(model):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            date=[],
            onpromotion=[],
            store_cluster=[],
            family=[],
            events=[],
        )
    ).to_csv(model, index=False)

# loading the trained model 
model= pickle.load(open('C:\LP4\Embedded-ML-model-into-Gui\ml\Time series dec_tree_model.pkl', 'rb'))

prediction(model)

# Setup execution
ml_components_dict = model(prediction)

labels = ml_components_dict['labels']
num_cols = ml_components_dict['num_cols']
cat_cols = ml_components_dict['cat_cols']
num_imputer = ml_components_dict['num_imputer']
cat_imputer = ml_components_dict['cat_imputer']
scaler = ml_components_dict['scaler']
encoder = ml_components_dict['encoder']
model = ml_components_dict['model']


st.image("https://ultimahoraec.com/wp-content/uploads/2021/03/dt.common.streams.StreamServer-768x426.jpg")
st.title("Sales predictor app")

st.caption("This app predicts sales patterns of Cooperation Favorita over time in different stores in Ecuador.")

form = st.form(key="information", clear_on_submit=True)

with form:

    cols = st.columns((1, 1))
    date = cols[1].date_input("date")
    onpromotion = cols[0].selectbox("onpromotion:", ["Yes", "No"])
    store_cluster = cols[0].text_input("store_cluster")
    family = cols[1].selectbox("family:", ["AUTOMOTIVE", "BEAUTY AND FASHION", "BEVERAGES AND LIQUOR", "FROZEN FOODS", "Grocery", "HOME AND KITCHEN", "HOME CARE AND GARDEN", "PET SUPPLIES", "SCHOOL AND OFFICE SUPPLIES"], index=2)
    events = st.selectbox("events:", ["Holiday", "No holiday"])
    cols = st.columns(2)

    submitted = st.form_submit_button(label="Predict")

if submitted:
    st.success("Thanks!")
    pd.read_csv(model).append(
        dict(
            date=date,
            onpromotion=onpromotion,
            store_cluster=store_cluster,
            family=family,
            events=events,
        ),
        ignore_index=True,
    ).to_csv(model, index=False)
    st.balloons()

expander = st.expander("See all records")
with expander:
    df = pd.read_csv(model)
    st.dataframe(df)

