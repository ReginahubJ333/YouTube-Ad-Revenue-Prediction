
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# LOAD MODEL & FEATURES

model_data = pickle.load(open("linear_model.pkl", "rb"))

coefficients = np.array(model_data["coefficients"])
intercept = model_data["intercept"]
feature_columns = model_data["features"]

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
        'upload_year': upload_year, # Corrected from 'year'
        'upload_month': upload_month, # Corrected from 'month'
        'upload_day': upload_day,
        'upload_hour': upload_hour,
        'upload_weekday': upload_weekday,
        # Initialize one-hot encoded columns to 0
        'Engagement_Rate': 0, # Will be calculated later if needed, or derived from input
        'Interaction': 0,     # Will be calculated later if needed, or derived from input
        'RPM': 0,             # Will be calculated later if needed, or derived from input
        'category_Entertainment': 0,
        'category_Gaming': 0,
        'category_Lifestyle': 0,
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

    # Calculate engineered features
    input_dict["Engagement_Rate"] = (input_dict["likes"] + input_dict["comments"]) / input_dict["views"]
    input_dict["Interaction"] = input_dict["likes"] + input_dict["comments"]
    # RPM is ad_revenue_usd / views * 1000, which is the target, so we don't calculate it here for input
    # It's an output of the model essentially, or a different metric of performance.
    # For prediction input, we should only include features that are actually used by the model

    # one-hot mapping for categorical fields
    if "category_" + category in feature_columns:
        input_dict["category_" + category] = 1
    if "device_" + device in feature_columns:
        input_dict["device_" + device] = 1
    if "country_" + country in feature_columns:
        input_dict["country_" + country] = 1

    # Ensure all feature_columns are present and in the correct order
    final_input = {col: [input_dict.get(col, 0)] for col in feature_columns}
    final_df = pd.DataFrame(final_input)

    # Recalculate Engagement_Rate and Interaction after setting other values
    # and ensure RPM is not passed as an input feature if it's derived from the target
    if 'Engagement_Rate' in final_df.columns:
        final_df['Engagement_Rate'] = (final_df['likes'] + final_df['comments']) / final_df['views']
    if 'Interaction' in final_df.columns:
        final_df['Interaction'] = final_df['likes'] + final_df['comments']
    if 'RPM' in final_df.columns: # Remove RPM if it's in feature_columns but should not be an input
        final_df = final_df.drop(columns=['RPM'])

    # Reorder columns to match the model's expected feature order
    final_df = final_df[feature_columns]

    return final_df


# PREDICTION

# Prepare input dataframe
input_df = prepare_input()

# Prediction
if st.button("Predict Ad Revenue"):
    input_df = prepare_input()   

    prediction = np.dot(
        input_df.values,
        coefficients
    ) + intercept

    st.success(f"Estimated Ad Revenue: ${prediction[0]:.2f}")

