#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:35:58 2026

@author: surindersinghchhabra
"""

import numpy as np  # Importing numpy for numerical operations


def calculate_distance(parallax):
    """Calculate distances in parsecs using parallax in milliarcseconds.

    Formula:
        distance (pc) = 1000 / parallax (mas)
    """
    return 1000 / parallax


def calculate_silhouette_per_cluster(features, labels):
    """Calculate the silhouette score for each cluster.

    The silhouette score measures how similar a point is to its own cluster
    (cohesion) compared to other clusters (separation). Higher is better.

    Parameters:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)

    Returns:
        cluster_scores: dict mapping cluster label -> average silhouette score
    """
    unique_labels = np.unique(labels)
    cluster_scores = {}

    for cluster_label in unique_labels:
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_silhouette_scores = []

        for i in cluster_indices:
            # Points in the same cluster as point i
            same_cluster = features[labels == labels[i]]

            # Intra-cluster distances
            intra_distances = np.linalg.norm(same_cluster - features[i], axis=1)

            # Average distance to other points in same cluster (exclude itself)
            if len(intra_distances) > 1:
                a_i = np.mean(intra_distances[intra_distances > 0])
            else:
                a_i = 0

            # Find nearest other cluster (min mean distance)
            b_i = float("inf")
            for other_label in unique_labels:
                if other_label == labels[i]:
                    continue

                other_cluster = features[labels == other_label]
                inter_distances = np.linalg.norm(other_cluster - features[i], axis=1)
                b_i = min(b_i, np.mean(inter_distances))

            # Silhouette for point i
            denom = max(a_i, b_i)
            if denom > 0:
                s_i = (b_i - a_i) / denom
            else:
                s_i = 0

            cluster_silhouette_scores.append(s_i)

        cluster_scores[cluster_label] = np.mean(cluster_silhouette_scores)

    return cluster_scores
