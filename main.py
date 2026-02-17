#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:18:40 2026

@author: surindersinghchhabra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custom_kmeans import CustomKMeans
from bayesian import (
    normalize_data,
    perform_bayesian_clustering,
    calculate_log_likelihood,
)
from silhouttescores import calculate_distance, calculate_silhouette_per_cluster


# Function to calculate distance from parallax in parsecs
def calculate_distance_from_parallax(parallax):
    """Calculate distances in parsecs using parallax in milliarcseconds."""
    return 1000 / parallax  # Convert parallax to distance in parsecs


# Function to perform K-means clustering and calculate WCSS (within-cluster sum of squares)
def perform_kmeans_and_evaluate(data, features, n_clusters, tolerance):
    # Initialize and fit the CustomKMeans algorithm with specified number of clusters and tolerance
    kmeans = CustomKMeans(n_clusters=n_clusters, tol=tolerance)
    kmeans.fit(features)

    # Calculate WCSS: sum of squared distances to centroids
    wcss = np.sum(
        [
            np.sum((features[kmeans.labels == cluster] - kmeans.centroids[cluster]) ** 2)
            for cluster in range(n_clusters)
        ]
    )

    # Add predicted cluster labels to the original dataset for further analysis
    data["cluster"] = kmeans.labels

    # Obtain statistics for each cluster (e.g., centroid, count of points, etc.)
    cluster_stats = kmeans.get_cluster_stats()
    stats_df = pd.DataFrame.from_dict(cluster_stats, orient="index")

    # Save the statistics to a CSV file for future reference
    stats_df.to_csv(
        f"kmeans_cluster_stats_tol_{tolerance}.csv",
        index_label="Cluster",
    )

    # 3D Visualization of clusters and centroids
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each cluster's points in 3D space
    for cluster in range(n_clusters):
        cluster_data = features[kmeans.labels == cluster]
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            cluster_data[:, 2],
            label=f"Cluster {cluster}",
        )

    # Plot centroids in red
    ax.scatter(
        kmeans.centroids[:, 0],
        kmeans.centroids[:, 1],
        kmeans.centroids[:, 2],
        c="red",
        marker="x",
        s=100,
        label="Centroids",
    )

    # Set titles and labels for the axes
    ax.set_title(f"K-Means Clustering (Tolerance: {tolerance})")
    ax.set_xlabel("RA (Normalized)")
    ax.set_ylabel("Dec (Normalized)")
    ax.set_zlabel("Distance (Normalized)")
    ax.legend()

    plt.show()

    print(f"K-Means clustering complete for tolerance {tolerance}. WCSS: {wcss}")
    return wcss


def main():
    # Load Gaia dataset (a file containing star data from the Gaia mission)
    try:
        gaia_data = pd.read_csv("gaia_data.csv")  # Replace with the actual file path
    except FileNotFoundError:
        print(
            "Error: 'gaia_data.csv' not found. Please ensure the dataset is in the correct directory."
        )
        return

    # Calculate distance from parallax for each star and select relevant features
    gaia_data["distance"] = calculate_distance_from_parallax(gaia_data["parallax"])
    features = gaia_data[["ra", "dec", "distance"]].dropna().values

    # Normalize the features (RA, Dec, Distance)
    normalized_features = normalize_data(features)

    # ---- K-Means experiments (vary tolerance) ----
    tolerances = [1e-1, 1e-3, 1e-5, 1e-7]
    kmeans_wcss = {}

    for tol in tolerances:
        wcss = perform_kmeans_and_evaluate(
            gaia_data,
            normalized_features,
            n_clusters=5,
            tolerance=tol,
        )
        kmeans_wcss[tol] = wcss

    # ---- Bayesian clustering experiments (vary components) ----
    n_components_values = [3, 5, 7, 9]
    bayesian_log_likelihood = {}

    for n_components in n_components_values:
        centroids, labels = perform_bayesian_clustering(
            normalized_features,
            n_clusters=n_components,
            tolerance=1e-6,
        )

        log_likelihood = calculate_log_likelihood(
            normalized_features,
            centroids,
            labels,
            n_clusters=n_components,
        )

        bayesian_log_likelihood[n_components] = log_likelihood

        # 3D Visualization for Bayesian Clustering
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for cluster in range(n_components):
            cluster_data = normalized_features[labels == cluster]
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                cluster_data[:, 2],
                label=f"Cluster {cluster}",
            )

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="red",
            marker="x",
            s=100,
            label="Centroids",
        )

        ax.set_title(f"Bayesian Clustering (Components: {n_components})")
        ax.set_xlabel("RA (Normalized)")
        ax.set_ylabel("Dec (Normalized)")
        ax.set_zlabel("Distance (Normalized)")
        ax.legend()

        plt.show()

        print(
            f"Bayesian clustering complete for {n_components} components. "
            f"Log-Likelihood: {log_likelihood}"
        )

    # ---- Best configs by WCSS / log-likelihood ----
    best_kmeans_tol = min(kmeans_wcss, key=kmeans_wcss.get)
    best_kmeans_wcss = kmeans_wcss[best_kmeans_tol]

    best_bayesian_components = max(bayesian_log_likelihood, key=bayesian_log_likelihood.get)
    best_bayesian_ll = bayesian_log_likelihood[best_bayesian_components]

    print(
        f"Best K-Means configuration: Tolerance = {best_kmeans_tol} with WCSS = {best_kmeans_wcss}"
    )
    print(
        f"Best Bayesian (GMM) configuration: Components = {best_bayesian_components} "
        f"with Log-Likelihood = {best_bayesian_ll}"
    )

    print("Comparison of Methods:")
    print(f"K-Means WCSS: {best_kmeans_wcss}")
    print(f"Bayesian Log-Likelihood: {best_bayesian_ll}")

    # NOTE: This comparison mixes metrics (WCSS vs log-likelihood).
    # Leaving your logic as-is, but you may want a consistent comparison method.
    if -best_kmeans_wcss > best_bayesian_ll:
        print("K-Means performed better for cluster identification.")
    else:
        print("Bayesian (GMM) performed better for cluster identification.")

    # ---- Silhouette analysis for both methods ----
    kmeans_silhouettes = {}
    for tol in tolerances:
        kmeans = CustomKMeans(n_clusters=5, tol=tol)
        kmeans.fit(normalized_features)

        silhouette_scores = calculate_silhouette_per_cluster(
            normalized_features,
            kmeans.labels,
        )
        kmeans_silhouettes[tol] = silhouette_scores

    bayesian_silhouettes = {}
    for n_components in n_components_values:
        centroids, labels = perform_bayesian_clustering(
            normalized_features,
            n_clusters=n_components,
            tolerance=1e-6,
        )

        silhouette_scores = calculate_silhouette_per_cluster(
            normalized_features,
            labels,
        )
        bayesian_silhouettes[n_components] = silhouette_scores

    print("K-Means Silhouette Scores:")
    for tol, silhouette_scores in kmeans_silhouettes.items():
        print(f"Tolerance {tol}:")
        for cluster, score in silhouette_scores.items():
            print(f"  Cluster {cluster}: {score:.4f}")

    print("\nBayesian Clustering Silhouette Scores:")
    for n_components, silhouette_scores in bayesian_silhouettes.items():
        print(f"Components {n_components}:")
        for cluster, score in silhouette_scores.items():
            print(f"  Cluster {cluster}: {score:.4f}")

    kmeans_avg_silhouettes = {
        tol: np.mean(list(scores.values())) for tol, scores in kmeans_silhouettes.items()
    }
    bayesian_avg_silhouettes = {
        n_components: np.mean(list(scores.values()))
        for n_components, scores in bayesian_silhouettes.items()
    }

    best_kmeans_tol = max(kmeans_avg_silhouettes, key=kmeans_avg_silhouettes.get)
    best_bayesian_components = max(
        bayesian_avg_silhouettes, key=bayesian_avg_silhouettes.get
    )

    print("\nBest Configurations:")
    print(
        f"K-Means: Tolerance = {best_kmeans_tol} with Average Silhouette = "
        f"{kmeans_avg_silhouettes[best_kmeans_tol]:.4f}"
    )
    print(
        f"Bayesian Clustering: Components = {best_bayesian_components} with Average Silhouette = "
        f"{bayesian_avg_silhouettes[best_bayesian_components]:.4f}"
    )

    if kmeans_avg_silhouettes[best_kmeans_tol] > bayesian_avg_silhouettes[best_bayesian_components]:
        print("K-Means performed better overall.")
    else:
        print("Bayesian Clustering performed better overall.")


if __name__ == "__main__":
    main()
