# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "statsmodels",
#     "scikit-learn",
#     "missingno",
#     "python-dotenv",
#     "requests",
#     "seaborn",
# ]
# ///

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

# Setup logging for debugging and runtime information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file, such as API tokens
load_dotenv()

# Fetch AI Proxy token from .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Check if token is available; if not, exit the script
if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API request to communicate with the AI service
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

# Function to request AI to generate the narrative story for the dataset analysis
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Construct the prompt for the AI to generate a narrative about the dataset
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

    # Send the request to the AI service and handle any potential errors
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    # Return the generated story content from the AI response
    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

# Function to read CSV with automatic encoding detection
def read_csv_with_encoding(file_path):
    """Try to read a CSV file with different encodings."""
    try:
        # Attempt to read using UTF-8 encoding
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, use chardet to detect the encoding
        rawdata = open(file_path, 'rb').read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        logging.info(f"Detected encoding: {encoding} for file {file_path}")
        return pd.read_csv(file_path, encoding=encoding)

# Function to subset large datasets based on size
def subset_large_dataset(data, max_size_kb=1000):
    """
    Subset the dataset if it exceeds the specified size (in KB).
    By default, it will subset if the file is larger than 1000 KB (1 MB).
    """
    file_size = data.memory_usage(index=True).sum() / 1024  # Get file size in KB
    logging.info(f"Dataset size: {file_size:.2f} KB")

    # If dataset size exceeds the max_size_kb threshold, subset it
    if file_size > max_size_kb:
        logging.info(f"Dataset exceeds {max_size_kb} KB. Subsetting to a smaller sample.")
        # Randomly select a subset of the data (e.g., 10%)
        data = data.sample(frac=0.1, random_state=42)
        logging.info(f"Subsetted dataset to {len(data)} rows.")
    
    return data

# Perform basic statistical analysis on the dataset
def basic_analysis(data):
    # Generate a summary, missing value count, and data type information
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Outlier detection using Interquartile Range (IQR)
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)  # Select numeric columns
    Q1 = numeric_data.quantile(0.25)  # Calculate the first quartile (25%)
    Q3 = numeric_data.quantile(0.75)  # Calculate the third quartile (75%)
    IQR = Q3 - Q1  # Interquartile Range
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()  # Find outliers
    return {"outliers": outliers}

# Save the plot and close the figure to free memory
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')  # Save with tight bounding box
    plt.close(fig)  # Close the figure after saving
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

# Enhanced Correlation matrix with better readability
def generate_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    if numeric_data.empty:
        logging.warning("No numeric columns for correlation matrix.")
        return None
    corr = numeric_data.corr()  # Calculate the correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))  # Create figure with larger size
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)  # Create heatmap
    ax.set_title("Correlation Matrix", fontsize=16)  # Set title for the plot
    return save_plot(fig, "correlation_matrix")

# Enhanced PCA plot for dimensionality reduction
def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()  # Drop missing values for PCA
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    pca = PCA(n_components=2)  # Reduce to two dimensions for visualization
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))  # Scale and apply PCA
    fig, ax = plt.subplots(figsize=(10, 8))  # Create figure
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax, palette="viridis")  # Scatter plot for PCA results
    ax.set_title("PCA Plot", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    return save_plot(fig, "pca_plot")

# Enhanced DBSCAN clustering plot with better color palette
def dbscan_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()  # Drop missing values for clustering
    if numeric_data.empty:
        logging.warning("No numeric data for DBSCAN.")
        return None
    scaler = StandardScaler()  # Standardize the data for DBSCAN
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Initialize DBSCAN
    clusters = dbscan.fit_predict(scaled_data)  # Perform clustering
    numeric_data['cluster'] = clusters  # Add cluster column
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="coolwarm", ax=ax)  # Plot clusters
    ax.set_title("DBSCAN Clustering", fontsize=16)
    return save_plot(fig, "dbscan_clusters")

# Hierarchical clustering dendrogram with clearer labels
def hierarchical_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()  # Drop missing values for clustering
    if numeric_data.empty:
        logging.warning("No numeric data for hierarchical clustering.")
        return None
    linked = linkage(numeric_data, 'ward')  # Perform hierarchical clustering
    fig, ax = plt.subplots(figsize=(12, 8))  # Create figure
    dendrogram(linked, ax=ax)  # Create dendrogram
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=16)
    return save_plot(fig, "hierarchical_clustering")

# Save README file with detailed structure and analysis results
def save_readme(content):
    try:
        readme_path = "README.md"
        with open(readme_path, "w") as f:
            f.write(content)  # Write content to README file
        logging.info(f"README saved in the current directory.")
    except Exception as e:
        logging.error(f"Error saving README: {e}")
        sys.exit(1)

# Full analysis workflow: processes each file and generates visualizations, narrative, and README
def analyze_and_generate_output(file_paths):
    for file_path in file_paths:
        try:
            logging.info(f"Reading file: {file_path}")
            data = read_csv_with_encoding(file_path)  # Read the dataset
            
            # Handle large datasets by subsetting them if necessary
            data = subset_large_dataset(data)
            
            # Perform basic analysis on the dataset
            analysis = basic_analysis(data)
            outliers = outlier_detection(data)
            combined_analysis = {**analysis, **outliers}

            image_paths = []
            # Generate visualizations
            image_paths.append(generate_correlation_matrix(data))
            image_paths.append(generate_pca_plot(data))
            image_paths.append(dbscan_clustering(data))
            image_paths.append(hierarchical_clustering(data))

            # Generate the narrative based on the analysis and visualizations
            narrative = get_ai_story(str(combined_analysis), str(data.info()), image_paths)

            # Save the visualizations
            for image_path in image_paths:
                if image_path:
                    logging.info(f"Visualization saved: {image_path}")

            # Prepare and save the README file
            readme_content = f"""
            # Automated Data Analysis Report

            ## Dataset Summary:
            {analysis}

            ## Visualizations:
            ![Correlation Matrix](correlation_matrix.png)
            ![PCA Plot](pca_plot.png)
            ![DBSCAN Clustering](dbscan_clusters.png)
            ![Hierarchical Clustering](hierarchical_clustering.png)

            ## Generated Narrative:
            {narrative}
            """
            save_readme(readme_content)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

# Main function to parse arguments and start analysis
def main():
    parser = argparse.ArgumentParser(description="Automated Data Analysis Script")
    parser.add_argument('file_paths', metavar='dataset.csv', type=str, nargs='+', help="Paths to CSV datasets for analysis")
    args = parser.parse_args()
    
    analyze_and_generate_output(args.file_paths)

# Run the main function
if __name__ == "__main__":
    main()

