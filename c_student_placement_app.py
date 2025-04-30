
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Placement_Data.csv")
    df = df.dropna()
    return df

df = load_data()

# Encode categorical columns
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.drop('status')

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable separately (don't overwrite original labels)
target_encoder = LabelEncoder()
df['status_encoded'] = target_encoder.fit_transform(df['status'])

# Features and target
X = df.drop(['status', 'status_encoded'], axis=1)
y = df['status_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("Student Placement Prediction System")

st.subheader("Model Performance")
st.metric("Accuracy on Test Data", f"{acc:.2%}")
st.text(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# User Input Section
st.subheader("Enter Student Data for Prediction")
user_input = {}

for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        selected = st.selectbox(col, options)
        user_input[col] = label_encoders[col].transform([selected])[0]
    else:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

if st.button("Predict Placement Status"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]
    st.success(f"Predicted Placement Status: {predicted_label}")

# Data Visualization Section
st.subheader("Data Visualizations")

if st.checkbox("Correlation Heatmap"):
    st.write("Correlation between numerical features:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', ax=ax)
    st.pyplot(fig)

if st.checkbox("Placement Status Distribution"):
    st.write("Count of Placed vs Not Placed:")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='status', data=df, ax=ax2)
    st.pyplot(fig2)

st.caption("Developed by Lavi Pawar")
