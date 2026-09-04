# Clustering Analysis of Used Fiat 500 Cars in Italy

An unsupervised machine learning project applying K-Means and Hierarchical clustering to segment the used Fiat 500 car market in Italy, identifying distinct market segments to support pricing and inventory strategy.

## Introduction

Used car listings vary widely in price, mileage, age, and engine specs. This project groups listings into distinct market segments using clustering, so that patterns in the used Fiat 500 market can be identified without predefined labels.

## Dataset

The dataset (`automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv`) contains used Fiat 500 listings scraped from an Italian automotive marketplace, with attributes such as price, age, mileage (km), and engine power.

## Methodology

**Exploratory Data Analysis & Preprocessing**
- Inspected data types, non-null counts, and descriptive statistics
- Checked for missing values
- Visualised pairwise relationships between features with a pair plot
- Standardised features using `StandardScaler` so no single feature dominates the distance calculation

**K-Means Clustering**
- Used the elbow method (inertia vs. number of clusters, k = 1 to 10) to select an appropriate number of clusters
- Fitted K-Means with k = 6 clusters (`k-means++` initialisation)
- Reduced dimensions to 2 components via PCA for visualisation, and plotted the resulting clusters as a labelled scatter plot

**Hierarchical Clustering**
- Built a dendrogram (Ward linkage) to visually confirm the optimal number of clusters
- Applied Agglomerative Clustering with 6 clusters (Euclidean distance, Ward linkage)
- Visualised the resulting clusters as a scatter plot for comparison against the K-Means result

## Tech Stack

- **Python** — `pandas`, `numpy`, `scikit-learn` (KMeans, AgglomerativeClustering, PCA, StandardScaler), `scipy` (dendrogram), `matplotlib`, `seaborn`

## Repository Contents

- `task2_fiat500_clustering.ipynb` — full notebook: EDA, preprocessing, K-Means clustering (elbow method + PCA visualisation), and Hierarchical clustering (dendrogram + comparison)
- `report.pdf` — full report including business benefits, insights, recommendations, and references

## What I'd Improve Next

- Quantitatively compare K-Means and Hierarchical clustering results (e.g. silhouette score) rather than relying on visual comparison alone
- Profile each cluster's average price, mileage, and age to translate segments into concrete business recommendations
