import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load datasets
df1 = pd.read_csv("user_behavior_dataset.csv")
df2 = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df3 = pd.read_csv("survey.csv")

# Select columns
df1 = df1[['Screen On Time (hours/day)']]
df2 = df2[['Sleep Duration', 'Stress Level']]
df3 = df3[['treatment']]

# Merge
df = pd.concat([df1, df2, df3], axis=1)

# Clean
df['Screen On Time (hours/day)'] = pd.to_numeric(df['Screen On Time (hours/day)'], errors='coerce')
df['Sleep Duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')
df['Stress Level'] = pd.to_numeric(df['Stress Level'], errors='coerce')
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

df = df.dropna()

# Features
df['high_screen_time'] = (df['Screen On Time (hours/day)'] > 6).astype(int)
df['sleep_deficit'] = 8 - df['Sleep Duration']

# Target
df['burnout'] = (
    df['Screen On Time (hours/day)'] +
    df['Stress Level'] -
    df['Sleep Duration']
)

df['burnout'] = df['burnout'].apply(lambda x: 1 if x > 5 else 0)

# Model
X = df[['Screen On Time (hours/day)', 'Sleep Duration', 'Stress Level', 'treatment', 'high_screen_time', 'sleep_deficit']]
y = df['burnout']

model = RandomForestClassifier()
model.fit(X, y)

# UI
st.title("Burnout Prediction")

screen_time = st.slider("Screen Time", 0.0, 12.0, 6.0)
sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
stress = st.slider("Stress Level", 1, 10, 5)
treatment = st.selectbox("Mental Health Support", ["No", "Yes"])

treatment_val = 1 if treatment == "Yes" else 0

if st.button("Predict"):
    input_data = [[screen_time, sleep, stress, treatment_val, int(screen_time>6), 8-sleep]]
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("High Burnout Risk")
    else:
        st.success("Low Burnout Risk")