import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

# ----------------------------------------
# 1. Page Configuration & Title
# ----------------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 AI-Powered Customer Segmentation Dashboard")
st.markdown("""
This application uses the **K-Means Clustering** algorithm to automatically group customers into distinct segments 
based on their demographic and spending behaviors. Tweak the sidebar controls to explore the data!
""")

# ----------------------------------------
# 2. Sidebar Controls
# ----------------------------------------
st.sidebar.header("🛠️ Dashboard Settings")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset (Optional)", type=["csv"])

# Hyperparameter Selection
st.sidebar.subheader("Algorithm Parameters")
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)

# Feature Selection
st.sidebar.subheader("Select Features for Clustering")
# Default placeholder features (will be overwritten if a file is uploaded)
all_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# ----------------------------------------
# 3. Data Loading Logic
# ----------------------------------------
@st.cache_data # Caches data so it doesn't regenerate on every slider move
def load_mock_data():
    """Generates a synthetic customer dataset similar to the classic Mall Customers data"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate 3 distinct customer groups artificially to ensure good clustering
    group1 = np.random.multivariate_normal([25, 20, 80], [[16, 5, 5], [5, 25, 5], [5, 5, 36]], int(n_samples*0.3))
    group2 = np.random.multivariate_normal([45, 60, 50], [[25, 5, 5], [5, 100, 10], [5, 10, 49]], int(n_samples*0.4))
    group3 = np.random.multivariate_normal([35, 90, 15], [[36, 5, 5], [5, 64, 5], [5, 5, 25]], int(n_samples*0.3))
    
    data = np.vstack([group1, group2, group3])
    df = pd.DataFrame(data, columns=all_features)
    # Clean up bounds and data types
    df["Age"] = df["Age"].clip(18, 70).astype(int)
    df["Annual Income (k$)"] = df["Annual Income (k$)"].clip(15, 130).astype(int)
    df["Spending Score (1-100)"] = df["Spending Score (1-100)"].clip(1, 100).astype(int)
    df.insert(0, 'CustomerID', range(1001, 1001 + n_samples))
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Successfully loaded uploaded dataset!")
    # Let user pick columns if they upload their own data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect("Features to cluster", numeric_cols, default=numeric_cols[:3])
else:
    df = load_mock_data()
    selected_features = all_features
    st.sidebar.info("Using auto-generated mock customer dataset.")

# ----------------------------------------
# 4. Clustering Pipeline (ML Logic)
# ----------------------------------------
if len(selected_features) >= 2:
    # Extract features and scale them (Crucial step for K-Means distance calculations)
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means Model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].apply(lambda x: f"Segment {x+1}") # Make cluster labels user-friendly
    
    # Calculate Silhouette Score performance metric
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    
    # ----------------------------------------
    # 5. Dashboard Layout & Visualizations
    # ----------------------------------------
    
    # Row 1: Key Performance Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers Analyzed", len(df))
    with col2:
        st.metric("Target Segments (K)", n_clusters)
    with col3:
        st.metric("Model Silhouette Score", f"{sil_score:.3f}", help="Closer to 1 means clusters are highly distinct and well-separated.")
        
    st.markdown("---")
    
    # Row 2: Charts
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.subheader("3D Cluster Separation")
        # If we have 3 features, plot in interactive 3D
        if len(selected_features) >= 3:
            fig_3d = px.scatter_3d(
                df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                color='Cluster', hover_data=['CustomerID'],
                title="Customer Groups in 3D Space",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # Fallback to standard 2D scatter plot if user selected only 2 features
            fig_2d = px.scatter(
                df, x=selected_features[0], y=selected_features[1],
                color='Cluster', hover_data=['CustomerID'],
                title="Customer Groups in 2D Space",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    with col_chart2:
        st.subheader("Segment Distribution")
        # Pie chart showing breakdown percentages
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(
            cluster_counts, values='Count', names='Cluster', 
            title="Percentage of Customers per Segment",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    
    # Row 3: Deep Dive Analytics Table
    st.subheader("🔍 Segment Profiling & Data Summary")
    
    # Calculate average metric values per segment to let users interpret what the clusters mean
    summary_df = df.groupby('Cluster')[selected_features].mean().reset_index()
    summary_df.insert(1, 'Customer Count', df['Cluster'].value_counts().sort_index().values)
    
    st.markdown("**Average Characteristics per Segment:**")
    st.dataframe(summary_df.style.format(precision=1), use_container_width=True)
    
    # Expandable view to see raw underlying data rows
    with st.expander("📄 View Raw Customer Table"):
        st.dataframe(df, use_container_width=True)

else:
    st.error("Please select at least 2 features in the sidebar to perform clustering calculations.")