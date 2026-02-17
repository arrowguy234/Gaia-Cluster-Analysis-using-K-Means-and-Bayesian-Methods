#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:36:27 2026

@author: surindersinghchhabra
"""

import numpy as np


class CustomKMeans:
    """
    Simple K-Means implementation.

    Parameters:
        n_clusters (int): number of clusters
        max_iter (int): maximum iterations
        tol (float): convergence tolerance for centroid movement
    """

    def __init__(self, n_clusters=5, max_iter=300, tol=1e-7):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        self.centroids = None
        self.labels = None
        self.cluster_stats = None

    def fit(self, X):
        """Fit K-Means to data X."""
        np.random.seed(42)

        # Step 1: randomly initialize centroids from data points
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        # Step 2–4: assign clusters, update centroids, check convergence
        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)

            new_centroids = np.array(
                [
                    X[self.labels == k].mean(axis=0)
                    if np.any(self.labels == k)
                    else self.centroids[k]
                    for k in range(self.n_clusters)
                ]
            )

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.cluster_stats = self._calculate_cluster_stats(X)
        return self

    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = np.array(
            [[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X]
        )
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """Predict cluster labels for new points X."""
        distances = np.array(
            [[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X]
        )
        return np.argmin(distances, axis=1)

    def _calculate_cluster_stats(self, X):
        """Compute stats for each cluster."""
        stats = {}

        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]

            if len(cluster_points) > 0:
                intra_distances = np.linalg.norm(cluster_points - self.centroids[k], axis=1)
                stats[k] = {
                    "size": len(cluster_points),
                    "mean_distance_to_centroid": float(intra_distances.mean()),
                    "centroid": self.centroids[k],
                }
            else:
                stats[k] = {
                    "size": 0,
                    "mean_distance_to_centroid": np.nan,
                    "centroid": self.centroids[k],
                }

        return stats

    def get_cluster_stats(self):
        """Return computed cluster statistics."""
        return self.cluster_stats
