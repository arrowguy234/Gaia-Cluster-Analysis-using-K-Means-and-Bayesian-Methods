# Stellar Cluster Distance Estimation using Bayesian and K-Means Clustering

This project leverages custom implementations of **K-Means clustering**
and **Bayesian Gaussian Mixture Models (GMM)** to analyze and estimate
distances to stellar clusters using data from the **Gaia mission**.

The project includes preprocessing steps, clustering algorithms,
evaluation metrics, and visualization techniques for understanding
stellar cluster properties.

------------------------------------------------------------------------

## Installation

1.  Clone the repository:

``` bash
git clone https://github.com/arrowguy234/K_cluster-and-bayesian.git
cd K_cluster-and-bayesian
```

2.  Ensure you have **Python 3.8 or higher** installed.

------------------------------------------------------------------------

## Usage

### 1️⃣ Data Preparation

-   Place your dataset in the `data/` directory in CSV format.

-   Ensure the dataset contains the following columns:

    -   `parallax` --- Parallax values (milliarcseconds)
    -   `ra` --- Right Ascension
    -   `dec` --- Declination

You can run:

``` bash
python Gaia_visualisation.py
```

This generates database-related plots and statistical summaries.

------------------------------------------------------------------------

### 2️⃣ Running Clustering Algorithms

#### 🔹 K-Means Clustering

``` bash
python custom_kmeans.py
```

#### 🔹 Bayesian Clustering (GMM-style)

``` bash
python bayesian.py
```

This will:

-   Add Bayesian and K-Means cluster assignments to the dataset
-   Generate 3D cluster visualizations

------------------------------------------------------------------------

### 3️⃣ Silhouette Scores

``` bash
python silhouttescores.py
```

This prepares functions used to calculate silhouette scores inside
`main.py`.

------------------------------------------------------------------------

### 4️⃣ Main Execution File

``` bash
python main.py
```

This will:

-   Generate clustering plots (K-Means & Bayesian)
-   Print WCSS and log-likelihood values
-   Save clustering outputs to CSV files
-   Print silhouette scores
-   Compare clustering performance

------------------------------------------------------------------------

## Code Overview

### 📌 1. `custom_kmeans.py`

**CustomKMeans Class**

Implements a basic K-Means algorithm with cluster statistics.

Key methods:

-   `fit(X)` --- Fit clustering model to data
-   `predict(X)` --- Predict clusters for new data
-   `get_cluster_stats()` --- Retrieve cluster statistics (size,
    intra-cluster distance, centroids)

------------------------------------------------------------------------

### 📌 2. `bayesian.py`

Key functions:

-   `calculate_distance(parallax)` --- Compute distance from parallax
-   `normalize_data(X)` --- Scale features to \[0, 1\]
-   `perform_bayesian_clustering(...)` --- EM-style clustering
    implementation
-   `calculate_log_likelihood(...)` --- Evaluate clustering quality

------------------------------------------------------------------------

### 📌 3. `main.py`

-   Entry point for the project
-   Runs clustering experiments
-   Evaluates performance
-   Prints comparison metrics

------------------------------------------------------------------------

## Example Workflow

### 1. Normalize data and compute distances

``` python
from bayesian import calculate_distance, normalize_data

data['distance'] = calculate_distance(data['parallax'])
features = normalize_data(data[['ra', 'dec', 'distance']].values)
```

### 2. Perform K-Means clustering

``` python
from custom_kmeans import CustomKMeans

kmeans = CustomKMeans(n_clusters=5, tol=1e-7)
kmeans.fit(features)
print(kmeans.get_cluster_stats())
```

### 3. Perform Bayesian clustering

``` python
from bayesian import perform_bayesian_clustering

centroids, labels = perform_bayesian_clustering(features, n_clusters=5)
```

------------------------------------------------------------------------

## Output Details

### 🔹 K-Means Outputs

-   Cluster assignments
-   Cluster centroids
-   Intra-cluster distances
-   Cluster sizes

### 🔹 Bayesian Outputs

-   Log-likelihood values
-   Cluster centroids
-   3D visualizations

------------------------------------------------------------------------

## Dependencies

-   Python 3.8+
-   numpy
-   pandas
-   matplotlib

Install dependencies:

``` bash
pip install numpy pandas matplotlib
```

------------------------------------------------------------------------

## Contact

**Surinder Singh Chhabra**\
Email: surinder.chhabra0000@gmail.com
