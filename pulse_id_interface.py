import streamlit as st
import pandas as pd
import openai
import boto3
from io import StringIO
from datetime import datetime
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- OpenAI API Setup ---
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# --- Streamlit UI ---
st.title("🧠 RFM Segmentation with LLM & Clustering")

# --- Data Source Selection ---
data_source = st.radio("Choose Data Source", ["Upload CSV", "Fetch from S3"])

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
elif data_source == "Fetch from S3":
    bucket = 'pulseid-ai'
    key = 'Sagemaker/Visa Japan/transactions/AUTHORIZATION/2025/04_all_cleaned_combined/full_combined.csv'
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df_raw = pd.read_csv(obj['Body'])

# --- Step 1: Column Mapping via GPT ---
def map_columns_with_gpt(columns):
    prompt = f"""
    These are the columns in the uploaded dataset: {columns}

    Please check if we can calculate Recency, Frequency, and Monetary (RFM) values.
    - Recency needs a user_id and a transaction date.
    - Frequency needs a user_id.
    - Monetary needs a user_id and amount.

    If possible, map the uploaded columns to the expected ones:
    - user_id => external_user_id
    - txn_date => transactionDate
    - amount_spent => Monetary

    Respond in JSON format like:
    {{
      "can_calculate_rfm": true,
      "mapped_columns": {{
        "external_user_id": "user_id",
        "transactionDate": "txn_date",
        "Monetary": "amount_spent"
      }}
    }}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return eval(response.choices[0].message.content)

if 'df_raw' in locals():
    mapping_result = map_columns_with_gpt(df_raw.columns.tolist())

    if not mapping_result.get("can_calculate_rfm"):
        st.error("❌ Provided columns are not sufficient for RFM analysis.")
        st.stop()
    else:
        col_map = mapping_result["mapped_columns"]
        df_raw.rename(columns=col_map, inplace=True)
        st.success("✅ Required columns mapped successfully!")
        st.dataframe(df_raw.head())

        # --- Date Input ---
        input_date = st.date_input("Select Current Date", value=datetime.today())

        # --- RFM Calculation ---
        df_raw['transactionDate'] = pd.to_datetime(df_raw['transactionDate'].astype(str).str.split(' ').str[0], errors='coerce')
        df_raw['Monetary'] = pd.to_numeric(df_raw['Monetary'], errors='coerce')
        df_raw = df_raw.dropna(subset=['transactionDate', 'Monetary'])

        grouped = df_raw.groupby('external_user_id').agg(
            Frequency=('external_user_id', 'count'),
            Monetary=('Monetary', 'sum'),
            Recency=('transactionDate', 'max')
        ).reset_index()

        grouped['Recency'] = (input_date - grouped['Recency'].dt.date).dt.days

        # --- Elbow Method ---
        scaler = StandardScaler()
        scaled_rfm = scaler.fit_transform(grouped[['Recency', 'Frequency', 'Monetary']])
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled_rfm)
            sse.append(kmeans.inertia_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 11)), y=sse, mode='lines+markers'))
        fig.update_layout(title="Elbow Curve for Optimal k", xaxis_title="k", yaxis_title="SSE")
        st.plotly_chart(fig)

        # --- Select k ---
        k_selected = st.slider("Select number of clusters (k)", 2, 10, value=5)
        kmeans_final = KMeans(n_clusters=k_selected, random_state=42)
        grouped['Cluster'] = kmeans_final.fit_predict(scaled_rfm)

        # --- Cluster Summaries ---
        st.subheader("📊 Cluster Summary")
        for i in range(k_selected):
            cluster_i = grouped[grouped['Cluster'] == i]
            st.metric(label=f"Cluster {i}", value=f"{cluster_i.shape[0]} users")

        # --- Cluster Naming Button ---
        if st.button("🧠 Generate Meaningful Cluster Names"):
            cluster_names = {}
            for i in range(k_selected):
                cluster = grouped[grouped['Cluster'] == i][['Recency', 'Frequency', 'Monetary']]
                stats = cluster.describe().to_dict()
                prompt = f"""
                For cluster {i}, here are the stats:
                - Avg Recency: {stats['Recency']['mean']:.1f}
                - Avg Frequency: {stats['Frequency']['mean']:.1f}
                - Avg Monetary: {stats['Monetary']['mean']:.1f}
                Suggest a short business-friendly name.
                """
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )
                cluster_names[f"Cluster {i}"] = response.choices[0].message.content.strip()

            st.write("### 🏷️ Cluster Labels")
            st.json(cluster_names)

        # --- Show Final DataFrame ---
        st.subheader("🧾 Final RFM with Clusters")
        st.dataframe(grouped.head())
