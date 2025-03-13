import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Function to load dataset
@st.cache_data
def load_data():
    customers_ndf = pd.read_csv("customers_dataset.csv")
    orders_ndf = pd.read_csv("orders_dataset_cleaned.csv")
    order_items_ndf = pd.read_csv("order_items_dataset_cleaned.csv")
    geolocation_ndf = pd.read_csv("geolocation_dataset_cleaned.csv")
    return customers_ndf, orders_ndf, order_items_ndf, geolocation_ndf

# Function to convert date columns
def convert_datetime_columns(df, date_columns):
    """
    Convert string date columns to datetime type
    """
    df_copy = df.copy()
    
    for column in date_columns:
        if column in df_copy.columns:
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
    
    return df_copy

# Process orders dataset
def process_orders_dataset(orders_df):
    # Date columns in orders dataset
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    
    # Convert date columns
    orders_df = convert_datetime_columns(orders_df, date_columns)
    
    # Extract date components
    orders_df['purchase_year'] = orders_df['order_purchase_timestamp'].dt.year
    orders_df['purchase_month'] = orders_df['order_purchase_timestamp'].dt.month
    orders_df['purchase_day'] = orders_df['order_purchase_timestamp'].dt.day
    orders_df['purchase_dayofweek'] = orders_df['order_purchase_timestamp'].dt.dayofweek
    orders_df['purchase_hour'] = orders_df['order_purchase_timestamp'].dt.hour
    
    # Calculate delivery time (days)
    mask = (~orders_df['order_delivered_customer_date'].isna() & 
            ~orders_df['order_purchase_timestamp'].isna())
    
    orders_df.loc[mask, 'delivery_time_days'] = (
        orders_df.loc[mask, 'order_delivered_customer_date'] - 
        orders_df.loc[mask, 'order_purchase_timestamp']
    ).dt.total_seconds() / (24 * 60 * 60)
    
    return orders_df

# Process order items dataset
def process_order_items_dataset(order_items_df):
    # Date columns in order_items dataset
    date_columns = ['shipping_limit_date']
    
    # Convert date columns
    order_items_df = convert_datetime_columns(order_items_df, date_columns)
    
    # Calculate total item price (including freight)
    order_items_df['total_price'] = order_items_df['price'] + order_items_df['freight_value']
    
    return order_items_df

# Create RFM data
def create_rfm_data(customers_df, orders_df, order_items_df):
    if customers_df.empty or orders_df.empty or order_items_df.empty:
        st.error("Salah satu dataset kosong! Pastikan semua data tersedia.")
        return None, None
    
    # Konversi tanggal ke datetime
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    
    # Hitung total harga per transaksi (menghindari duplikasi)
    order_items_df['total_price'] = order_items_df['price'] + order_items_df['freight_value']
    
    # Gabungkan order dengan item details (hindari duplikasi!)
    order_data = orders_df.merge(order_items_df, on='order_id', how='inner')
    
    # Pastikan hanya pesanan yang selesai (delivered)
    completed_orders = order_data[order_data['order_status'] == 'delivered']
    
    # Hitung snapshot_date (tanggal terakhir transaksi + 1 hari)
    snapshot_date = completed_orders['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # Hitung RFM
    rfm = completed_orders.groupby('customer_id').agg(
        recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),  # Hari sejak terakhir transaksi
        frequency=('order_id', 'nunique'),  # Jumlah unik transaksi
        monetary=('total_price', 'sum')  # Total belanja
    ).reset_index()
    
    # Gabungkan dengan data pelanggan
    rfm = rfm.merge(customers_df[['customer_id']], on='customer_id', how='left')
    
    return rfm, snapshot_date

# Segment customers based on RFM scores
def segment_customers(rfm):
    if rfm is None or rfm.empty:
        print("Data RFM kosong! Tidak bisa melakukan segmentasi.")
        return None
    
    # Normalisasi skor 0-4
    rfm['R_score'] = pd.qcut(rfm['recency'], 10, labels=range(9, -1, -1))  # Skala 9-0
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 10, labels=range(0, 10))  # Skala 0-9
    rfm['M_score'] = pd.qcut(rfm['monetary'], 10, labels=range(0, 10))  # Skala 0-9

    # Buat RFM Score
    rfm['RFM_score'] = rfm[['R_score', 'F_score', 'M_score']].astype(str).agg(''.join, axis=1)

    # Mapping segmentasi yang lebih luas
    segment_map = {
        '444': 'Champions',
        '443': 'Champions',
        '442': 'Champions',
        '441': 'Champions',
        '434': 'Loyal Customers',
        '433': 'Loyal Customers',
        '432': 'Loyal Customers',
        '431': 'Loyal Customers',
        '424': 'Potential Loyalists',
        '423': 'Potential Loyalists',
        '422': 'Potential Loyalists',
        '421': 'Potential Loyalists',
        '414': 'New Customers',
        '413': 'New Customers',
        '412': 'New Customers',
        '411': 'Recent Customers',
        '344': 'Loyal Customers',
        '343': 'Loyal Customers',
        '342': 'Loyal Customers',
        '341': 'Loyal Customers',
        '334': 'Potential Loyalists',
        '333': 'Potential Loyalists',
        '332': 'Potential Loyalists',
        '331': 'Promising',
        '324': 'At Risk',
        '323': 'At Risk',
        '322': 'At Risk',
        '321': 'Need Attention',
        '314': 'Need Attention',
        '313': 'Need Attention',
        '312': 'About to Sleep',
        '311': 'About to Sleep',
        '244': 'At Risk',
        '243': 'At Risk',
        '242': 'At Risk',
        '241': 'At Risk',
        '234': 'Need Attention',
        '233': 'Need Attention',
        '232': 'About to Sleep',
        '231': 'About to Sleep',
        '224': 'Hibernating',
        '223': 'Hibernating',
        '222': 'Hibernating',
        '221': 'Hibernating',
        '214': 'Lost',
        '213': 'Lost',
        '212': 'Lost',
        '211': 'Lost',
        '144': 'Hibernating',
        '143': 'Hibernating',
        '142': 'Hibernating',
        '141': 'Hibernating',
        '134': 'Lost',
        '133': 'Lost',
        '132': 'Lost',
        '131': 'Lost',
        '124': 'Lost',
        '123': 'Lost',
        '122': 'Lost',
        '121': 'Lost',
        '114': 'Lost',
        '113': 'Lost',
        '112': 'Lost',
        '111': 'Lost',
        '000': 'Lost',
        '001': 'Lost',
        '002': 'Lost',
        '003': 'Lost',
        '004': 'Lost'
    }

    # Tambahkan default kategori
    def assign_segment(score):
        if score in segment_map:
            return segment_map[score]
        else:
            r, f, m = int(score[0]), int(score[1]), int(score[2])
            if r >= 3 and f >= 3 and m >= 3:
                return "Loyal Customers"
            elif r >= 3 and f >= 2 and m >= 2:
                return "Potential Loyalists"
            elif r <= 1 and f <= 1 and m <= 1:
                return "Lost"
            elif r <= 2 and f <= 2 and m <= 2:
                return "Hibernating"
            elif r <= 2 and f >= 3:
                return "At Risk"
            else:
                return "Others"

    rfm['segment'] = rfm['RFM_score'].apply(assign_segment)

    return rfm

# Prepare geospatial data
def prepare_geo_data(orders_df, customers_df, geolocation_df):
    # Merge orders and customers
    order_customer = orders_df.merge(customers_df, on='customer_id')
    
    # Group by zip code and state to get order counts
    zip_order_count = order_customer.groupby(['customer_zip_code_prefix', 'customer_state']).agg({
        'order_id': 'count'
    }).reset_index()
    
    zip_order_count.columns = ['zip_code_prefix', 'state', 'order_count']
    
    # Get unique geolocation data (to avoid duplicates)
    unique_geolocation = geolocation_df.drop_duplicates(subset=['geolocation_zip_code_prefix'])
    
    # Merge with geolocation data
    geo_data = zip_order_count.merge(
        unique_geolocation,
        left_on='zip_code_prefix',
        right_on='geolocation_zip_code_prefix',
        how='left'
    )
    
    # Filter out rows with NaN coordinates
    geo_data = geo_data.dropna(subset=['geolocation_lat', 'geolocation_lng'])
    
    return geo_data

# Create orders heatmap
def create_order_map(geo_data):
    # Create base map centered on Brazil
    m = folium.Map(
        location=[-14.235, -51.9253],
        zoom_start=4,
        tiles='CartoDB positron'
    )
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each location
    for idx, row in geo_data.iterrows():
        folium.CircleMarker(
            location=[row['geolocation_lat'], row['geolocation_lng']],
            radius=5,
            popup=f"State: {row['state']}<br>Orders: {row['order_count']}",
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(marker_cluster)
    
    # Add heatmap layer
    heat_data = [[row['geolocation_lat'], row['geolocation_lng'], row['order_count']] 
                for idx, row in geo_data.iterrows()]
    
    HeatMap(heat_data, radius=15).add_to(m)
    
    return m

# Function to show Overview page
def show_overview_page(customers_df, orders_df, order_items_df):
    st.title("E-Commerce Data Overview")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{customers_df['customer_id'].nunique():,}")
    
    with col2:
        st.metric("Total Orders", f"{orders_df['order_id'].nunique():,}")
    
    with col3:
        total_revenue = order_items_df['price'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col4:
        avg_order_value = order_items_df.groupby('order_id')['price'].sum().mean()
        st.metric("Average Order Value", f"${avg_order_value:.2f}")
    
    # Order status distribution
    st.subheader("Order Status Distribution")
    status_counts = orders_df['order_status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.pie(status_counts, values='Count', names='Status', 
                 title='Order Status Distribution',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)
    
    # Monthly order trend
    st.subheader("Monthly Order Trend")
    orders_df['month_year'] = orders_df['order_purchase_timestamp'].dt.strftime('%Y-%m')
    monthly_orders = orders_df.groupby('month_year').size().reset_index(name='count')
    
    fig = px.line(monthly_orders, x='month_year', y='count', 
                  title='Monthly Order Trend',
                  labels={'month_year': 'Month-Year', 'count': 'Number of Orders'})
    st.plotly_chart(fig)
    
    # Customer distribution by state
    st.subheader("Customer Distribution by State")
    state_counts = customers_df['customer_state'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    
    fig = px.bar(state_counts.head(10), x='State', y='Count', 
                 title='Top 10 States by Customer Count',
                 color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig)

# Function to show RFM Analysis page
def show_rfm_analysis_page(customers_df, orders_df, order_items_df):
    st.title("Customer Segmentation - RFM Analysis")
    
    with st.spinner('Calculating RFM metrics...'):
        rfm_data, snapshot_date = create_rfm_data(customers_df, orders_df, order_items_df)
        
        if rfm_data is None:
            return
        
        rfm_segmented = segment_customers(rfm_data)
    
    if rfm_segmented is None:
        return

    st.success(f"RFM Analysis completed as of {snapshot_date.strftime('%Y-%m-%d')}")
    
    # RFM Metrics Summary
    st.subheader("RFM Metrics Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Recency (days)", f"{rfm_segmented['recency'].mean():.1f}")
    
    with col2:
        st.metric("Average Frequency (orders)", f"{rfm_segmented['frequency'].mean():.1f}")
    
    with col3:
        st.metric("Average Monetary Value", f"${rfm_segmented['monetary'].mean():.2f}")
    
    # Segment Distribution
    st.subheader("Customer Segment Distribution")
    segment_counts = rfm_segmented['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    fig = px.bar(segment_counts, x='Segment', y='Count', 
                 title='Customer Segments',
                 color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    # RFM Metrics by Segment
    st.subheader("RFM Metrics by Segment")
    rfm_by_segment = rfm_segmented.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    rfm_by_segment.columns = ['Segment', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary Value', 'Customer Count']
    rfm_by_segment = rfm_by_segment.sort_values('Customer Count', ascending=False)
    
    st.dataframe(rfm_by_segment)

    # Compare Segments
    st.subheader("Compare Segments")
    segments_to_compare = st.multiselect(
        "Select segments to compare:",
        options=rfm_segmented['segment'].unique(),
        default=rfm_segmented['segment'].value_counts().nlargest(3).index.tolist()
    )

    if segments_to_compare:
        filtered_rfm = rfm_segmented[rfm_segmented['segment'].isin(segments_to_compare)]
        fig = px.scatter_3d(filtered_rfm, x='recency', y='frequency', z='monetary',
                            color='segment', hover_name='customer_id')
        st.plotly_chart(fig)

# Function to show Geospatial Analysis page
def show_geospatial_analysis_page(orders_df, customers_df, geolocation_df):
    st.title("Geospatial Analysis of Orders")
    
    # Prepare geospatial data
    with st.spinner('Preparing geospatial data...'):
        geo_data = prepare_geo_data(orders_df, customers_df, geolocation_df)
    
    st.success(f"Geospatial data prepared for {len(geo_data)} locations")
    
    # Orders by state
    st.subheader("Order Distribution by State")
    state_orders = geo_data.groupby('state')['order_count'].sum().reset_index()
    state_orders = state_orders.sort_values('order_count', ascending=False)
    
    fig = px.bar(state_orders, x='state', y='order_count',
                 title='Orders by State',
                 labels={'state': 'State', 'order_count': 'Number of Orders'},
                 color='order_count', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    # Interactive map
    st.subheader("Interactive Order Heatmap")
    m = create_order_map(geo_data)
    folium_static(m, width=1000, height=600)
    
    # Top cities by order count
    st.subheader("Top Cities by Order Count")
    city_orders = geo_data.groupby(['geolocation_city', 'state'])['order_count'].sum().reset_index()
    city_orders = city_orders.sort_values('order_count', ascending=False)
    
    st.dataframe(city_orders.head(20))

# Main function
def main():
    # Add a header for the entire app
    st.sidebar.image("https://img.icons8.com/color/48/000000/shopping-cart--v2.png", width=80)
    st.sidebar.title("E-Commerce Dashboard")
    
    # Load data
    customers_df, orders_df, order_items_df, geolocation_df = load_data()
    
    # Process data
    orders_df = process_orders_dataset(orders_df)
    order_items_df = process_order_items_dataset(order_items_df)
    
    # Create sidebar navigation
    st.sidebar.subheader("Navigation")
    menu = st.sidebar.radio(
        "Select Analysis:",
        ("Overview", "RFM Analysis", "Geospatial Analysis")
    )
    
    # Display the selected page
    if menu == "Overview":
        show_overview_page(customers_df, orders_df, order_items_df)
    elif menu == "RFM Analysis":
        show_rfm_analysis_page(customers_df, orders_df, order_items_df)
    elif menu == "Geospatial Analysis":
        show_geospatial_analysis_page(orders_df, customers_df, geolocation_df)

if __name__ == "__main__":
    main()