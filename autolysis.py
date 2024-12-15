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
#     "tenacity",  # Added tenacity for retry mechanisms
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Setup logging with a standardized format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Fetch AI Proxy token from environment variables
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API requests
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def get_ai_story(dataset_summary, dataset_info, visualizations):
    """
    Request AI to generate a narrative story based on dataset analysis.

    Args:
        dataset_summary (dict): Summary statistics of the dataset.
        dataset_info (dict): Information about the dataset's columns and data types.
        visualizations (dict): Paths to generated visualization images.

    Returns:
        str: Generated narrative or an error message if failed.
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset vividly. What does the data represent? What are its features? What is the significance of this data? Create a compelling story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used. Highlight techniques like missing value handling, outlier detection, clustering, and dimensionality reduction (PCA). How do these methods provide insights?
    3. **Key Insights and Discoveries**: What were the major findings? What trends or patterns emerged that can be interpreted as discoveries? Were there any unexpected results?
    4. **Implications and Actions**: Discuss the implications of these findings. How do they influence decisions? What actionable recommendations would you provide based on the analysis?
    5. **Visualizations**: Describe the visualizations included. What do they reveal about the data? How do they complement the analysis and findings?

    **Dataset Summary**:
    {json.dumps(dataset_summary, indent=2)}

    **Dataset Info**:
    {json.dumps(dataset_info, indent=2)}

    **Visualizations**:
    {json.dumps(visualizations, indent=2)}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(Exception)
)
def load_data(file_path):
    """
    Load dataset from a CSV file with automatic encoding detection.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        SystemExit: If the file cannot be loaded.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the encoding of the file
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded with {encoding} encoding.")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise  # Trigger retry

def basic_analysis(data):
    """
    Perform basic statistical analysis on the dataset.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Summary statistics, missing values, and column information.
    """
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

def outlier_detection(data):
    """
    Detect outliers in the dataset using the IQR method.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        dict: Number of outliers detected per numeric column.
    """
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

def save_plot(fig, plot_name):
    """
    Save a matplotlib figure to a PNG file.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        plot_name (str): The base name for the saved plot file.

    Returns:
        str: Path to the saved plot image.
    """
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved as {plot_path}")
    return plot_path

def generate_correlation_matrix(data):
    """
    Generate and save a correlation matrix heatmap.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        str or None: Path to the saved heatmap image or None if no numeric data.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        logging.warning("No numeric columns for correlation matrix.")
        return None
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix", fontsize=16)
    return save_plot(fig, "correlation_matrix")

def generate_pca_plot(data):
    """
    Generate and save a PCA scatter plot.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        str or None: Path to the saved PCA plot image or None if insufficient data.
    """
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax, palette="viridis")
    ax.set_title("PCA Plot", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    return save_plot(fig, "pca_plot")

def dbscan_clustering(data):
    """
    Perform DBSCAN clustering and generate a scatter plot of clusters.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        str or None: Path to the saved DBSCAN clusters plot or None if no numeric data.
    """
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for DBSCAN.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x=numeric_data.iloc[:, 0],
        y=numeric_data.iloc[:, 1],
        hue='cluster',
        palette="coolwarm",
        ax=ax,
        legend='full'
    )
    ax.set_title("DBSCAN Clustering", fontsize=16)
    ax.set_xlabel(numeric_data.columns[0], fontsize=14)
    ax.set_ylabel(numeric_data.columns[1], fontsize=14)
    ax.legend(title='Cluster')
    return save_plot(fig, "dbscan_clusters")

def hierarchical_clustering(data):
    """
    Perform hierarchical clustering and generate a dendrogram.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        str or None: Path to the saved dendrogram image or None if no numeric data.
    """
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for hierarchical clustering.")
        return None
    linked = linkage(numeric_data, 'ward')
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linked, ax=ax, truncate_mode='level', p=5)
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=16)
    ax.set_xlabel("Sample Index", fontsize=14)
    ax.set_ylabel("Distance", fontsize=14)
    return save_plot(fig, "hierarchical_clustering")

def save_readme(content):
    """
    Save the generated narrative to a README.md file.

    Args:
        content (str): The narrative content to save.
    """
    try:
        readme_path = "README.md"
        with open(readme_path, "w", encoding='utf-8') as f:
            f.write(content)
        logging.info(f"README saved in the current directory as {readme_path}.")
    except Exception as e:
        logging.error(f"Error saving README: {e}")
        sys.exit(1)

def analyze_and_generate_output(file_path):
    """
    Perform the full analysis workflow and generate outputs.

    Args:
        file_path (str): Path to the input CSV dataset.

    Returns:
        tuple: Generated narrative and paths to visualization images.
    """
    # Load the dataset
    data = load_data(file_path)
    
    # Perform basic statistical analysis
    analysis = basic_analysis(data)
    
    # Detect outliers
    outliers = outlier_detection(data)
    
    # Combine analysis results
    combined_analysis = {**analysis, **outliers}

    # Generate visualizations and collect their paths
    image_paths = {
        'correlation_matrix': generate_correlation_matrix(data),
        'pca_plot': generate_pca_plot(data),
        'dbscan_clusters': dbscan_clustering(data),
        'hierarchical_clustering': hierarchical_clustering(data)
    }

    # Prepare data information for narrative generation
    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }

    # Generate the narrative using AI
    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    if not narrative:
        narrative = "Error: Narrative generation failed. Please verify the AI service."

    # Save the narrative to README.md
    save_readme(f"Dataset Analysis:\n\n{narrative}")

    return narrative, image_paths

def main():
    """
    Main entry point of the script. Parses command-line arguments and initiates analysis.
    """
    parser = argparse.ArgumentParser(description="Automated Data Analysis and Narrative Generation")
    parser.add_argument("file_path", type=str, help="Path to the dataset CSV file")
    args = parser.parse_args()

    analyze_and_generate_output(args.file_path)

if __name__ == "__main__":
    main()
