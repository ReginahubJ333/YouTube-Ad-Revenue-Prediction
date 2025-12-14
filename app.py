import streamlit as st
import pickle
import pandas as pd
import numpy as np

# LOAD MODEL & FEATURES

model = pickle.load(open("best_random_forest_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="wide")

st.title("YouTube Ad Revenue Prediction App")
st.write("Enter your video details below to estimate your revenue.")

# USER INPUT FORM

st.header("📌 Enter Video Metrics")

col1, col2 = st.columns(2)

with col1:
    views = st.number_input("Views", min_value=0, value=10000)
    likes = st.number_input("Likes", min_value=0, value=500)
    comments = st.number_input("Comments", min_value=0, value=100)
    watch_time = st.number_input("Watch Time (minutes)", min_value=1.0, value=15000.0)

with col2:
    video_length = st.number_input("Video Length (minutes)", min_value=1.0, value=10.0)
    subscribers = st.number_input("Channel Subscribers", min_value=0, value=50000)

    # categorical inputs
    category = st.selectbox(
        "Category",
        ["Education", "Gaming", "Entertainment", "Music", "Tech"]
    )
    device = st.selectbox(
        "Top Device",
        ["Mobile", "TV", "Tablet"]
    )
    country = st.selectbox(
        "Top Country",
        ["IN", "US", "UK", "CA", "DE"]
    )

# date features
st.header("📅 Upload Details")

upload_year = st.number_input("Upload Year", min_value=2020, max_value=2030, value=2024)
upload_month = st.number_input("Upload Month", min_value=1, max_value=12, value=9)
upload_day = st.number_input("Upload Day", min_value=1, max_value=31, value=15)
upload_hour = st.number_input("Upload Hour (0-23)", min_value=0, max_value=23, value=10)
upload_weekday = st.number_input("Upload Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

# PROCESS INPUT INTO MODEL FORMAT

def prepare_input():
    input_dict = {
        'views': views,
        'likes': likes,
        'comments': comments,
        'watch_time_minutes': watch_time,
        'video_length_minutes': video_length,
        'subscribers': subscribers,
        'upload_year': upload_year,
        'upload_month': upload_month,
        'upload_day': upload_day,
        'upload_hour': upload_hour,
        'upload_weekday': upload_weekday,
        'Engagement_Rate': 0,
        'Interaction': 0,
        'category_Entertainment': 0,
        'category_Gaming': 0,
        'category_Music': 0,
        'category_Tech': 0,
        'device_Mobile': 0,
        'device_TV': 0,
        'device_Tablet': 0,
        'country_CA': 0,
        'country_DE': 0,
        'country_IN': 0,
        'country_UK': 0,
        'country_US': 0
    }

    # engineered features
    input_dict["Engagement_Rate"] = (
        (input_dict["likes"] + input_dict["comments"]) / input_dict["views"]
        if input_dict["views"] > 0 else 0
    )
    input_dict["Interaction"] = input_dict["likes"] + input_dict["comments"]

    # one-hot encoding
    if "category_" + category in input_dict:
        input_dict["category_" + category] = 1
    if "device_" + device in input_dict:
        input_dict["device_" + device] = 1
    if "country_" + country in input_dict:
        input_dict["country_" + country] = 1

    final_df = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_columns}])
    return final_df

# PREDICTION

if st.button("Predict Revenue"):
    input_data = prepare_input()
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Ad Revenue: **${prediction:.2f} USD**")
