# Required libraries and setup
import os
import sys
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
from PIL import Image
import chardet  # To detect file encoding
from io import BytesIO
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Fetch AI Proxy token from .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

# Function to request AI to generate the narrative story
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset vividly. What does the data represent? What are its features? What is the significance of this data? Create a compelling story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used. Highlight techniques like missing value handling, outlier detection, clustering, and dimensionality reduction (PCA). How do these methods provide insights?
    3. **Key Insights and Discoveries**: What were the major findings? What trends or patterns emerged that can be interpreted as discoveries? Were there any unexpected results?
    4. **Implications and Actions**: Discuss the implications of these findings. How do they influence decisions? What actionable recommendations would you provide based on the analysis?
    5. **Visualizations**: Describe the visualizations included. What do they reveal about the data? How do they complement the analysis and findings?

    **Dataset Summary**:
    {dataset_summary}

    **Dataset Info**:
    {dataset_info}

    **Visualizations**:
    {visualizations}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

# Function to read CSV with automatic encoding detection
def read_csv_with_encoding(file_path):
    """Try to read a CSV file with different encodings."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        rawdata = open(file_path, 'rb').read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        logging.info(f"Detected encoding: {encoding} for file {file_path}")
        return pd.read_csv(file_path, encoding=encoding)

# Function to subset large datasets
def subset_large_dataset(data, max_size_kb=1000):
    """Subset the dataset if it exceeds the specified size."""
    file_size = data.memory_usage(index=True).sum() / 1024
    logging.info(f"Dataset size: {file_size:.2f} KB")

    if file_size > max_size_kb:
        logging.info(f"Dataset exceeds {max_size_kb} KB. Subsetting to a smaller sample.")
        data = data.sample(frac=0.1, random_state=42)
        logging.info(f"Subsetted dataset to {len(data)} rows.")
    
    return data

# Function to handle missing values and convert categorical data
def preprocess_data(data):
    # Handle missing values by filling with the median for numerical columns
    data = data.fillna(data.median())
    
    # Handle categorical columns by encoding them
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    
    return data

# Basic analysis function
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Outlier detection using IQR
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Function to save enhanced visualizations
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

# Enhanced correlation heatmap
def generate_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        logging.warning("No numeric columns for correlation matrix.")
        return None
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix", fontsize=16)
    return save_plot(fig, "correlation_matrix")

# PCA plot function
def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax, palette="viridis")
    ax.set_title("PCA Plot", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    return save_plot(fig, "pca_plot")

# Full analysis workflow with enhanced structure
def analyze_and_generate_output(file_paths):
    for file_path in file_paths:
        try:
            logging.info(f"Reading file: {file_path}")
            data = read_csv_with_encoding(file_path)
            data = subset_large_dataset(data)
            data = preprocess_data(data)

            analysis = basic_analysis(data)
            outliers = outlier_detection(data)
            combined_analysis = {**analysis, **outliers}

            image_paths = []
            image_paths.append(generate_correlation_matrix(data))
            image_paths.append(generate_pca_plot(data))

            narrative = get_ai_story(str(combined_analysis), str(data.info()), image_paths)

            for image_path in image_paths:
                if image_path:
                    logging.info(f"Visualization saved: {image_path}")

            readme_content = f"""
            # Automated Data Analysis Report

            ## Dataset Summary:
            {analysis}

            ## Visualizations:
            ![Correlation Matrix](correlation_matrix.png)
            ![PCA Plot](pca_plot.png)

            ## Generated Narrative:
            {narrative}
            """
            save_readme(readme_content)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Automated Data Analysis Script")
    parser.add_argument('file_paths', metavar='dataset.csv', type=str, nargs='+', help="Paths to CSV datasets for analysis")
    args = parser.parse_args()
    
    analyze_and_generate_output(args.file_paths)

if __name__ == "__main__":
    main()
