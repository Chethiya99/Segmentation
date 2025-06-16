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
import traceback

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        st.error("OpenAI API key not found. Please check your .env file")
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    st.stop()

st.set_page_config(page_title="RFM Segmentation Tool", layout="wide")
st.title("📊 RFM Segmentation with GPT-powered Column Mapping")

# --- Section 1: File Source ---
st.sidebar.header("📂 Data Source Options")
source_type = st.sidebar.radio("Choose CSV Source", ["Upload CSV", "S3 Bucket"])

df_raw = None

if source_type == "S3 Bucket":
    st.sidebar.subheader("S3 Configuration")
    bucket = st.sidebar.text_input("Bucket Name", value="pulseid-ai")
    key = st.sidebar.text_input("File Path", value="Sagemaker/Visa Japan/transactions/AUTHORIZATION/2025/04_all_cleaned_combined/full_combined.csv")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")
    load_button = st.sidebar.button("Load from S3")

    if load_button:
        try:
            s3 = boto3.client('s3', region_name=aws_region)
            obj = s3.get_object(Bucket=bucket, Key=key)
            df_raw = pd.read_csv(io.BytesIO(obj['Body'].read()), nrows=5000)
            st.success(f"Successfully loaded {len(df_raw)} records from S3")
        except Exception as e:
            st.error(f"Failed to load from S3: {str(e)}")
            st.error(traceback.format_exc())
            st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_raw)} records from uploaded file")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {str(e)}")
            st.stop()

if df_raw is None:
    st.info("Please upload a file or connect to S3 to begin")
    st.stop()

# Display raw data preview
with st.expander("🔍 Raw Data Preview"):
    st.dataframe(df_raw.head())

# --- Section 2: GPT Column Mapping ---
st.header("🔑 Column Mapping")

def map_columns_with_gpt(column_names):
    prompt = f"""You are a data analyst helping with RFM segmentation. The dataset has these columns:
{column_names}

Identify which columns should map to these RFM components:
1. user_id (customer identifier)
2. transaction_date (date of transaction)
3. Monetary (transaction amount)

Return ONLY a valid JSON object with the following structure using the actual column names:

{{
    "user_id": "column_name",
    "transaction_date": "column_name", 
    "Monetary": "column_name",
    "confidence": "high/medium/low"
}}

Important:
- The response must contain ONLY the JSON object
- If unclear about any mapping, set "confidence": "low"
- Do not include any additional text or explanation"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        # Extract and validate JSON from response
        response_content = response.choices[0].message.content
        if response_content.startswith("```json"):
            response_content = response_content[7:-3]  # Remove markdown json tags if present
        return response_content.strip()
    except Exception as e:
        return json.dumps({"error": f"GPT API error: {str(e)}"})

# Run GPT mapping when requested
if st.button("🤖 Auto-map Columns with GPT"):
    with st.spinner("Consulting GPT for column mapping..."):
        mapping_result = map_columns_with_gpt(df_raw.columns.tolist())
        
        try:
            column_map = json.loads(mapping_result)
            
            if "error" in column_map:
                st.error(column_map["error"])
            else:
                st.session_state.column_map = column_map
                st.success("GPT-generated mapping:")
                st.json(column_map)
                
                if column_map.get("confidence", "low") == "low":
                    st.warning("GPT has low confidence in this mapping. Please verify.")
                
        except json.JSONDecodeError:
            st.error("Failed to parse GPT response as JSON. Response was:")
            st.code(mapping_result)
        except Exception as e:
            st.error(f"Mapping error: {str(e)}")

# Manual mapping fallback
st.subheader("Manual Column Mapping")
if 'column_map' not in st.session_state:
    st.session_state.column_map = {
        "user_id": "",
        "transaction_date": "",
        "Monetary": ""
    }

# Create manual mapping interface
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.column_map["user_id"] = st.selectbox(
        "User/Customer ID Column",
        options=df_raw.columns,
        index=df_raw.columns.get_loc(st.session_state.column_map["user_id"]) 
        if st.session_state.column_map["user_id"] in df_raw.columns else 0
    )
with col2:
    st.session_state.column_map["transaction_date"] = st.selectbox(
        "Transaction Date Column",
        options=df_raw.columns,
        index=df_raw.columns.get_loc(st.session_state.column_map["transaction_date"])
        if st.session_state.column_map["transaction_date"] in df_raw.columns else 0
    )
with col3:
    st.session_state.column_map["Monetary"] = st.selectbox(
        "Transaction Amount Column",
        options=df_raw.columns,
        index=df_raw.columns.get_loc(st.session_state.column_map["Monetary"])
        if st.session_state.column_map["Monetary"] in df_raw.columns else 0
    )

# --- Section 3: RFM Calculation ---
st.header("📊 RFM Calculation")

try:
    # Prepare data with selected columns
    df_clean = df_raw.rename(columns={
        st.session_state.column_map['user_id']: 'user_id',
        st.session_state.column_map['transaction_date']: 'transaction_date',
        st.session_state.column_map['Monetary']: 'Monetary'
    }).copy()
    
    # Data cleaning
    rfm_df = df_clean[['user_id', 'transaction_date', 'Monetary']].copy()
    rfm_df['transaction_date'] = pd.to_datetime(rfm_df['transaction_date'], errors='coerce')
    rfm_df['Monetary'] = pd.to_numeric(rfm_df['Monetary'], errors='coerce')
    
    # Remove rows with invalid dates/amounts
    initial_count = len(rfm_df)
    rfm_df.dropna(subset=['transaction_date', 'Monetary'], inplace=True)
    final_count = len(rfm_df)
    
    if final_count == 0:
        st.error("No valid data remaining after cleaning")
        st.stop()
        
    if initial_count != final_count:
        st.warning(f"Removed {initial_count - final_count} rows with invalid data ({final_count} remaining)")

    # Calculate RFM metrics
    current_date = datetime.now().date()
    
    rfm = rfm_df.groupby('user_id').agg(
        Frequency=('user_id', 'count'),
        Monetary=('Monetary', 'sum'),
        Recency=('transaction_date', lambda x: (current_date - x.max().date()).days)
    ).reset_index()
    
    # Display results
    st.success(f"RFM metrics calculated for {len(rfm)} customers")
    with st.expander("View RFM Data"):
        st.dataframe(rfm.head())
    
    # Calculate percentiles for RFM scores with duplicate handling
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
    except ValueError as e:
        st.warning(f"Recency scoring adjustment needed: {str(e)}")
        rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5,4,3,2,1]).astype(int)
    
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    except ValueError as e:
        st.warning(f"Frequency scoring adjustment needed: {str(e)}")
        # Alternative approach when quintiles fail
        unique_freq = rfm['Frequency'].nunique()
        if unique_freq < 5:
            st.warning(f"Only {unique_freq} unique frequency values - using simpler binning")
            bins = [-1, 1, 2, 3, 5, float('inf')]
            labels = [1,2,3,4,5][:len(bins)-1]
            rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=bins, labels=labels).astype(int)
        else:
            rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=5, labels=[1,2,3,4,5]).astype(int)
    
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    except ValueError as e:
        st.warning(f"Monetary scoring adjustment needed: {str(e)}")
        rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=5, labels=[1,2,3,4,5]).astype(int)
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
except Exception as e:
    st.error(f"Error during RFM calculation: {str(e)}")
    st.error(traceback.format_exc())
    st.stop()

# --- Section 4: Clustering ---
st.header("🧮 Customer Segmentation")

# Elbow Method
st.subheader("Optimal Number of Clusters")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

sse = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

fig = px.line(x=list(K), y=sse, markers=True,
             labels={'x': 'Number of Clusters', 'y': 'Sum of Squared Errors'},
             title='Elbow Method for Optimal k')
st.plotly_chart(fig, use_container_width=True)

# Cluster selection
k_value = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)

# Apply clustering
kmeans = KMeans(n_clusters=k_value, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualize clusters
fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                   color='Cluster', hover_name='user_id',
                   title='3D Cluster Visualization')
st.plotly_chart(fig, use_container_width=True)

# --- Section 5: Cluster Analysis ---
st.header("📈 Cluster Analysis")

# Cluster summaries
for cluster_id in sorted(rfm['Cluster'].unique()):
    cluster_data = rfm[rfm['Cluster'] == cluster_id]
    
    with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} customers"):
        st.write(f"**Avg Recency:** {cluster_data['Recency'].mean():.1f} days")
        st.write(f"**Avg Frequency:** {cluster_data['Frequency'].mean():.1f} transactions")
        st.write(f"**Avg Monetary Value:** ${cluster_data['Monetary'].mean():,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(cluster_data.describe())
        with col2:
            fig = px.box(cluster_data, y=['Recency', 'Frequency', 'Monetary'])
            st.plotly_chart(fig, use_container_width=True)

# --- Section 6: GPT Cluster Naming ---
st.header("🏷️ Cluster Naming")

if st.button("✨ Generate Cluster Names with GPT"):
    with st.spinner("Generating meaningful cluster names..."):
        try:
            # Prepare cluster summaries with numpy types converted
            cluster_summaries = []
            for cluster_id in sorted(rfm['Cluster'].unique()):
                cluster_data = rfm[rfm['Cluster'] == cluster_id]
                summary = {
                    "cluster": int(cluster_id),  # Convert numpy to native Python int
                    "size": int(len(cluster_data)),
                    "recency_mean": float(cluster_data['Recency'].mean()),
                    "frequency_mean": float(cluster_data['Frequency'].mean()),
                    "monetary_mean": float(cluster_data['Monetary'].mean()),
                    "top_rfm_scores": {
                        str(k): int(v) for k, v in 
                        cluster_data['RFM_Score'].value_counts().head(3).items()
                    }
                }
                cluster_summaries.append(summary)
            
            prompt = f"""Analyze these customer clusters for RFM segmentation:
{json.dumps(cluster_summaries, indent=2, cls=NumpyEncoder)}

For each cluster, suggest:
1. A short descriptive name (e.g., "High Value Loyalists")
2. Key characteristics
3. Recommended engagement strategy

Return ONLY a valid JSON object in this format:
{{
    "clusters": [
        {{
            "id": 0,
            "name": "Cluster Name",
            "characteristics": "Key traits",
            "strategy": "Recommended actions"
        }}
    ]
}}

Important:
- The response must contain ONLY the JSON object
- Do not include any additional text or explanation"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Extract and validate JSON from response
            response_content = response.choices[0].message.content
            if response_content.startswith("```json"):
                response_content = response_content[7:-3]  # Remove markdown json tags if present
            
            cluster_names = json.loads(response_content.strip())
            st.session_state.cluster_names = cluster_names
            st.success("Generated cluster names:")
            st.json(cluster_names)
            
            # Apply names to dataframe
            name_mapping = {c['id']: c['name'] for c in cluster_names['clusters']}
            rfm['Cluster_Name'] = rfm['Cluster'].map(name_mapping)
            
        except Exception as e:
            st.error(f"Failed to generate cluster names: {str(e)}")
            st.error(traceback.format_exc())

# Display final results
if 'cluster_names' in st.session_state:
    st.subheader("Final Segmentation Results")
    st.dataframe(rfm.sort_values('Monetary', ascending=False))
    
    # Export options
    st.download_button(
        label="📥 Download Results as CSV",
        data=rfm.to_csv(index=False),
        file_name="rfm_segmentation.csv",
        mime="text/csv"
    )

# --- Section 7: Help/Instructions ---
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
    **RFM Segmentation Guide**
    
    1. **Upload Data**: Provide your transaction data via CSV upload or S3
    2. **Column Mapping**: Automatically map columns with GPT or manually select them
    3. **RFM Calculation**: The tool will calculate Recency, Frequency, Monetary values
    4. **Clustering**: Determine optimal clusters and visualize results
    5. **Analysis**: Review cluster characteristics and get AI-generated names
    
    **Key Concepts**:
    - **Recency**: How recently a customer purchased (lower = better)
    - **Frequency**: How often they purchase (higher = better)
    - **Monetary**: How much they spend (higher = better)
    """)
