📊 YouTube Ad Revenue Prediction using Machine Learning
🔍 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict YouTube Ad Revenue (USD) based on video performance metrics.
The solution includes data analysis, feature engineering, model training, hyperparameter tuning, and a Streamlit web application for real-time predictions.

The goal is to help creators estimate potential ad revenue before publishing a video.

🎯 Problem Statement

YouTube ad revenue depends on multiple factors such as:

Watch time
Views
Likes and comments
Subscriber count
Video category, device, and country
Manually estimating revenue is unreliable.
This project uses Machine Learning to provide accurate revenue predictions.

📁 Dataset Description

Records: 122,400
Features: 15 original → expanded to 32+ engineered features
Target Variable: ad_revenue_usd

Key Features:

views
likes
comments
watch_time_minutes
video_length_minutes
subscribers
category
device
country
Upload date and time features

🧹 Data Cleaning & Preprocessing

Handled missing values using median imputation
Removed duplicate records
Converted date column to datetime

Extracted:

upload_year
upload_month
upload_day
upload_hour
upload_weekday
One-hot encoded categorical features

📊 Exploratory Data Analysis (EDA)

Key insights from EDA:

Watch time has a near-perfect correlation with ad revenue (~0.99)
Views, likes, and comments show weak correlation
No significant outliers found
Dataset is clean and well-structured for modeling

🤖 Model Building

Three regression models were trained:

Model	RMSE	R² Score
Linear Regression	0.6118	0.9999
Decision Tree	0.3123	1.0000
Random Forest	0.1497	1.0000

➡ Random Forest Regressor was selected as the final model.

⚙️ Hyperparameter Tuning

Used RandomizedSearchCV with 150 combinations.

Best Parameters:
n_estimators = 500
max_depth = 40
max_features = 'sqrt'
min_samples_split = 5

💾 Model Export

The trained model and feature list were saved using pickle:
best_random_forest_model.pkl
feature_columns.pkl
These files are used by the Streamlit app for inference.

🖥️ Streamlit Web Application

The Streamlit app allows users to:

Enter video metrics
Select category, device, and country
Get instant ad revenue prediction

Run the app locally:
python -m streamlit run app.py

📦 Project Structure
youtube-ad-revenue-streamlit/
│── app.py
│── best_random_forest_model.pkl
│── feature_columns.pkl
│── requirements.txt
│── README.md

📜 Requirements

Install dependencies using:

pip install -r requirements.txt

🚀 Deployment

The application is deployed using Streamlit Community Cloud.

(Add your deployed app link here once live)

🔮 Future Enhancements

Integrate real YouTube API data
Add CPM and geographic multipliers
Deploy on cloud platforms like AWS / Render
Improve UI with visual analytics

⭐ Acknowledgements

Streamlit
Scikit-Learn
Pandas & NumPy
