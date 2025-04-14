import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.data_loader import load_data
from src.model import load_model
from src.predictor import predict_cluster



st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customer Segmentation")

menu = st.sidebar.radio("Navigate", ["📄 View Raw Data", "📊 Exploratory Data", "🔍 Identify Customer Cluster"])

# Load and prepare data
try:
    df = load_data("data/mall_customers.csv")
    df.drop(columns=['CustomerID'], inplace=True, errors='ignore')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

train_features = ["Age", "Annual_Income", "Spending_Score"]
predict_features = ["Age", "Annual_Income"]
k = 5

# Load model
try:
    model = load_model("models/kmeans_model.pkl")
    df["Cluster"] = model.predict(df[["Age", "Annual_Income", "Spending_Score"]])
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(df[["Age", "Annual_Income", "Spending_Score"]], df["Cluster"])
    inertia = model.inertia_
except Exception as e:
    st.error(f"Model loading or inference failed: {e}")
    st.stop()

# === Navigation ===
if menu == "📄 View Raw Data":
    st.subheader("📄 Raw Dataset")
    st.write(df)

elif menu == "📊 Exploratory Data":
    st.subheader("📊 Data Distributions")
    for feature in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    st.subheader("🔗 Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "🔍 Identify Customer Cluster":
    st.subheader("🧑 Identify Customer Cluster")

    col1, col2 = st.columns(2)
    with col1:
        age_input = st.number_input("Customer Age", min_value=1, max_value=100, value=30)
    with col2:
        income_input = st.number_input("Annual Income", min_value=1, max_value=200, value=60)

    if st.button("Show Cluster Details"):
        avg_score = df["Spending_Score"].mean()
        cluster_id = predict_cluster(model, age_input, income_input, avg_score)
        st.success(f"🔎 The customer belongs to **Cluster {cluster_id}**")

        st.subheader("📈 Clustering Metrics")
        st.write(f"**Inertia (WSS):** {inertia:.2f}")
        st.write(f"**Silhouette Score:** {silhouette:.2f}")

        st.subheader("📊 Cluster Counts")
        st.write(df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Counts'))

        st.subheader("🔍 Cluster Visualization (Age vs Annual Income)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Age", y="Annual_Income", hue='Cluster', palette='tab10', ax=ax)
        ax.scatter(age_input, income_input, color='black', s=100, label='New Customer', marker='X')
        ax.legend()
        ax.set_title("Customer Cluster Assignment")
        st.pyplot(fig)
