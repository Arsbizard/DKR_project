
# Fraudulent Bitcoin Transaction Detection Using Graph Machine Learning Techniques

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
    - [Transaction Class Distribution](#transaction-class-distribution)
    - [Temporal Analysis](#temporal-analysis)
3. [Graph Structure Analysis](#graph-structure-analysis)
    - [Transaction Network Visualization](#transaction-network-visualization)
    - [Node Degree Distribution](#node-degree-distribution)
4. [Feature Analysis](#feature-analysis)
    - [Feature Correlation Heatmap](#feature-correlation-heatmap)
    - [PCA of Features](#pca-of-features)
5. [GraphSAGE Model](#graphsage-model)
    - [Model Workflow](#model-workflow)
    - [Layer Aggregation](#layer-aggregation)
6. [Results and Analysis](#results-and-analysis)
    - [Node Embedding Visualization](#node-embedding-visualization)
    - [Prediction Scores Heatmap](#prediction-scores-heatmap)
7. [Conclusion and Future Work](#conclusion-and-future-work)

---

## Introduction

Bitcoin has revolutionized the financial sector by introducing decentralized digital transactions. However, it has also become a target for fraudulent activities, such as money laundering and illegal trades. Detecting fraudulent transactions in a Bitcoin network poses unique challenges due to its anonymous and decentralized nature.

This project addresses these challenges by leveraging Graph Machine Learning (ML) techniques, particularly the GraphSAGE model, to classify transactions as licit or illicit. By utilizing graph structures and node features, the project provides a robust way to identify suspicious activity in transaction networks.

---

## Dataset Overview

The Elliptic Bitcoin Dataset forms the backbone of this project. It consists of over 200,000 Bitcoin transactions represented as a directed graph. Each transaction is labeled as:
- **Licit**: Normal transactions.
- **Illicit**: Fraudulent transactions.
- **Unknown**: Transactions with no label.

### Transaction Class Distribution

The dataset is imbalanced, with most transactions being licit and a small proportion labeled as illicit. Understanding this distribution is crucial for addressing class imbalance in model training.

**Visualization**:
![Transaction Class Distribution](images/class_distribution_updated.png)

**Code**:
```python
# Load and analyze class distribution
classes_df = pd.read_csv('elliptic_txs_classes.csv')
class_counts = classes_df['class'].value_counts()

# Visualize class distribution
plt.figure(figsize=(8, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Updated Transaction Class Distribution")
plt.show()
```

---

### Temporal Analysis

Fraudulent activities often exhibit temporal patterns. Analyzing the frequency of transactions over time can help identify anomalies or bursts of illicit activity.

**Visualization**:
![Temporal Analysis](images/temporal_analysis_updated.png)

**Code**:
```python
# Analyze temporal trends
features_df = pd.read_csv('elliptic_txs_features.csv', header=None)
features_df = features_df.rename(columns={0: 'tx_id', 1: 'time_step'})
merged_df = pd.merge(features_df[['tx_id', 'time_step']], classes_df, left_on='tx_id', right_on='txId')

# Group by time step and visualize
time_class_counts = merged_df.groupby(['time_step', 'class']).size().unstack(fill_value=0)
time_class_counts.plot(kind='line', figsize=(12, 6))
plt.title('Updated Number of Transactions Over Time by Class')
plt.xlabel('Time Step')
plt.ylabel('Number of Transactions')
plt.legend(title='Class')
plt.show()
```

---

## Graph Structure Analysis

### Transaction Network Visualization

The graph structure of Bitcoin transactions reveals connectivity patterns. Fraud rings or tightly linked fraudulent nodes often form dense subgraphs, while licit transactions are more randomly distributed.

**Visualization**:
![Transaction Network Visualization](images/transaction_graph_updated.png)

**Code**:
```python
import networkx as nx

# Create and visualize the transaction graph
edges_df = pd.read_csv('elliptic_txs_edgelist.csv')
G = nx.from_pandas_edgelist(edges_df, source='txId1', target='txId2')

plt.figure(figsize=(12, 12))
nx.draw(G, node_color="lightblue", node_size=10, alpha=0.7)
plt.title("Updated Transaction Graph Structure")
plt.show()
```

---

### Node Degree Distribution

Nodes with unusually high degrees often act as transaction hubs, which can be either central fraud players or legitimate exchange points. Analyzing the degree distribution helps identify these influential nodes.

**Visualization**:
![Node Degree Distribution](images/degree_distribution_updated.png)

**Code**:
```python
# Calculate and visualize node degrees
degrees = [val for (node, val) in G.degree()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=50, color='blue', alpha=0.7)
plt.title("Updated Node Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.yscale('log')
plt.show()
```

---

## Feature Analysis

### Feature Correlation Heatmap

**Visualization**:
![Feature Correlation Heatmap](images/correlation_heatmap_updated.png)

**Code**:
```python
import seaborn as sns

# Compute and visualize feature correlations
corr_matrix = features_df.iloc[:, 2:].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Updated Feature Correlation Heatmap")
plt.show()
```

---

### PCA of Features

**Visualization**:
![PCA of Features](images/pca_scatter_updated.png)

**Code**:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Apply PCA and visualize
features_scaled = StandardScaler().fit_transform(features_df.iloc[:, 2:])
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['class'] = classes_df['class']

sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class', palette="deep")
plt.title("Updated PCA of Transaction Features")
plt.show()
```

---

## Conclusion and Future Work

This project demonstrates the effectiveness of Graph Machine Learning techniques, specifically GraphSAGE, for detecting fraudulent Bitcoin transactions. Future work could focus on:
- Expanding the dataset to improve generalization.
- Incorporating explainability methods for regulatory compliance.
- Adapting the model for real-time fraud detection.
