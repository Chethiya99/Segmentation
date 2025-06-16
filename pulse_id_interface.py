import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openai import OpenAI
import os
from dotenv import load_dotenv
import boto3
import io
import json

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="RFM Segmentation Tool", layout="wide")
st.title("📊 RFM Segmentation with GPT-powered Column Mapping")

# --- Section 1: File Source ---
st.sidebar.header("📂 Upload or S3")
source_type = st.sidebar.radio("Choose CSV Source", ["Upload CSV", "S3 Bucket"])

if source_type == "S3 Bucket":
    bucket = st.sidebar.text_input("S3 Bucket", value="pulseid-ai")
    key = st.sidebar.text_input("S3 Key", value="Sagemaker/Visa Japan/transactions/AUTHORIZATION/2025/04_all_cleaned_combined/full_combined.csv")
    load_button = st.sidebar.button("Load from S3")

    if load_button:
        try:
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=key)
            df_raw = pd.read_csv(io.BytesIO(obj['Body'].read()), nrows=5000)
            st.success("Loaded preview from S3")
        except Exception as e:
            st.error(f"Failed to load S3: {e}")
            st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully")
    else:
        st.stop()

# --- Section 2: GPT Column Mapping ---
def map_columns_with_gpt(column_names):
    prompt = f"""
You are a data analyst. The user uploaded a dataset with the following columns:
{column_names}

Your task is to check if the dataset contains or can be mapped to the required columns for RFM segmentation:
1. external_user_id or user_id
2. transaction_date
3. transaction_amount or Monetary

If the columns are available or can be mapped, return the mapping as a JSON dict:
{{"user_id": "external_user_id", "transaction_date": "transactionDate", "Monetary": "amount"}}

If not enough columns are available, respond with:
{{"error": "Provided columns are not enough"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

mapping_result = map_columns_with_gpt(df_raw.columns.tolist())

try:
    column_map = json.loads(mapping_result)
    if "error" in column_map:
        st.error(column_map["error"])
        st.stop()
except Exception:
    st.error("Failed to parse GPT mapping response")
    st.stop()

# --- Section 3: RFM Calculation ---
st.subheader("Preview of Cleaned Data")
df_raw.rename(columns={
    column_map['user_id']: 'user_id',
    column_map['transaction_date']: 'transaction_date',
    column_map['Monetary']: 'Monetary'
}, inplace=True)

# Clean & convert
rfm_df = df_raw[['user_id', 'transaction_date', 'Monetary']].copy()
rfm_df['transaction_date'] = pd.to_datetime(rfm_df['transaction_date'], errors='coerce')
rfm_df['Monetary'] = pd.to_numeric(rfm_df['Monetary'], errors='coerce')
rfm_df.dropna(subset=['transaction_date', 'Monetary'], inplace=True)

# Get current date
current_date = datetime.now().date()

# Group and calculate RFM
rfm = rfm_df.groupby('user_id').agg(
    Frequency=('user_id', 'count'),
    Monetary=('Monetary', 'sum'),
    Recency=('transaction_date', 'max')
).reset_index()
rfm['Recency'] = (current_date - rfm['Recency'].dt.date).dt.days
st.dataframe(rfm.head())

# --- Section 4: Elbow Curve ---
st.subheader("📈 Elbow Curve for Optimal k")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

sse = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

fig = px.line(x=list(K), y=sse, markers=True, labels={'x': 'k (Number of Clusters)', 'y': 'SSE'}, title='Elbow Curve')
st.plotly_chart(fig)

k_value = st.number_input("Select number of clusters (k)", min_value=2, max_value=10, value=3)

# --- Section 5: Final Clustering ---
kmeans = KMeans(n_clusters=k_value, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# --- Section 6: Cluster Overview ---
st.subheader("🧮 Cluster Summary")

for cluster_id in sorted(rfm['Cluster'].unique()):
    cluster_data = rfm[rfm['Cluster'] == cluster_id]
    st.markdown(f"### 🧊 Cluster {cluster_id} - ({len(cluster_data)} users)")
    st.dataframe(cluster_data.describe())

# --- Section 7: Generate Cluster Names using GPT ---
if st.button("🔍 Generate Meaningful Cluster Names with GPT"):
    cluster_insights = ""
    for cluster_id in sorted(rfm['Cluster'].unique()):
        cluster_data = rfm[rfm['Cluster'] == cluster_id][['Recency', 'Frequency', 'Monetary']]
        cluster_insights += f"\nCluster {cluster_id} Summary:\n"
        cluster_insights += str(cluster_data.describe()) + "\n"

    cluster_prompt = f"""
You are an expert in customer segmentation. Based on the following cluster summaries from RFM segmentation, assign meaningful business labels (like "High Value", "At Risk", "New Customers", etc.) to each cluster.
{cluster_insights}

Return as JSON: {{ "Cluster 0": "label0", "Cluster 1": "label1", ... }}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": cluster_prompt}],
        temperature=0
    )
    st.code(response.choices[0].message.content)
