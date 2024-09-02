# **Text Clustering and Visualization ON AG news Dataset**

## Overview

This repository contains code for text processing, clustering, and visualization using Python. The workflow involves cleaning and normalizing text data, generating embeddings with BERT, applying K-means clustering, reducing dimensionality, and visualizing the results using t-SNE and word clouds.

## 📥 Installation

To get started, you need to install the necessary Python libraries. These libraries are essential for text preprocessing, generating embeddings, clustering, and visualizing the results.

```bash
pip install nltk transformers torch scikit-learn matplotlib seaborn wordcloud
```

## 📚 Libraries and Tools

- **NLTK:** Used for text processing tasks such as tokenization, stopword removal, and lemmatization.
- **Transformers and Torch:** Leverage the BERT model to generate high-quality text embeddings.
- **Scikit-learn:** Provides tools for scaling data, dimensionality reduction (PCA, t-SNE), and clustering (K-means).
- **Matplotlib and Seaborn:** Used for creating visualizations like t-SNE scatter plots and silhouette scores.
- **WordCloud:** Generates word clouds to visually represent word frequencies in each cluster.

## 🚀 Workflow

### 1. **Data Preprocessing**

The preprocessing involves cleaning and normalizing text data to prepare it for further analysis.

- **Cleaning:** Removes punctuation, numbers, and stopwords.
- **Normalizing:** Lemmatizes words to their base forms.
- **Preprocessing Function:** Applies both cleaning and normalizing steps to the text data.

```python
def clean_text(text):
    # Implementation
def normalize_text(text):
    # Implementation
def preprocess_text(text):
    # Implementation
```

### 2. **Embedding Generation**

Precomputed BERT embeddings are loaded and standardized to ensure consistent scaling across features.

- **Hugging Face Dataset:** The dataset used for embedding generation is available on [Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news/viewer).
- **Embeddings File:** The precomputed embeddings can be downloaded from [this link](https://drive.google.com/file/d/1zCVZ5I56L6YNbOZHI05C2JPUK3fd4fVq/view?usp=sharing).
  
```python
train_embeddings = np.load('/path/to/train_embeddings.npy')
train_embeddings_scaled = StandardScaler().fit_transform(train_embeddings)
```

### 3. **Determining Optimal Number of Clusters**

The optimal number of clusters is determined using silhouette scores to ensure effective clustering.

```python
silhouette_scores = []
cluster_range = range(4, 8)
# Loop to calculate silhouette scores
```

### 4. **K-Means Clustering**

Apply K-means clustering with the optimal number of clusters to group similar text embeddings.

```python
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(train_embeddings_scaled)
```

### 5. **Dimensionality Reduction**

Dimensionality reduction is applied to make embeddings more manageable for visualization:

- **PCA:** Reduces dimensions to 50 while preserving variance.
- **t-SNE:** Further reduces dimensions to 2 for visualization.

```python
pca_result = PCA(n_components=50).fit_transform(train_embeddings_scaled)
tsne_result = TSNE(n_components=2, random_state=42).fit_transform(pca_result)
```

### 6. **Visualization**

- **t-SNE Plot:** Visualizes clusters in 2D to assess clustering quality.
- **Word Clouds:** Generates word clouds for each cluster to summarize frequent terms.

```python
def plot_wordcloud(text, title):
    # Implementation
```

### 7. **Cluster Analysis**

Prints details of each cluster, including the number of texts and sample texts for inspection.

```python
print("Cluster Details:")
# Implementation
```

## 🗂️ Dataset

The dataset used for processing is available on [Hugging Face](https://huggingface.co/datasets/fancyzhx/ag_news/viewer). The file used in this project is `train.parquet`, which contains text data for processing.

## 📊 Example Output

The output includes:

- t-SNE plots visualizing the clusters.
- Word clouds summarizing each cluster's content.
- Cluster details with sample texts and counts.

## 📝 Notes

- Ensure the path to the embeddings and dataset files is correct.
- Adjust `num_rows` to process a subset of data as needed.
- Modify the cluster range and other parameters according to your data and requirements.
