#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:36:19 2026

@author: surindersinghchhabra
"""

import numpy as np


# Normalize data without sklearn
def normalize_data(features):
    """Normalize the given dataset using min-max normalization.

    Args:
        features (numpy.ndarray): 2D array of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: normalized features scaled to [0, 1]
    """
    min_vals = np.min(features, axis=0)  # min per feature (column)
    max_vals = np.max(features, axis=0)  # max per feature (column)

    # Avoid division by zero if a column is constant
    denom = max_vals - min_vals
    denom[denom == 0] = 1

    return (features - min_vals) / denom


# Perform Bayesian clustering (basic EM-style loop using nearest-centroid assignment)
def perform_bayesian_clustering(features, n_clusters=3, tolerance=1e-6, max_iter=100):
    """Perform clustering using a basic Expectation-Maximization style loop.

    Steps:
        1) E-step: assign each point to the nearest centroid
        2) M-step: update centroids as means of assigned points

    Args:
        features (numpy.ndarray): shape (n_samples, n_features)
        n_clusters (int): number of clusters
        tolerance (float): convergence threshold based on centroid movement
        max_iter (int): maximum number of iterations

    Returns:
        (centroids, labels)
            centroids (numpy.ndarray): shape (n_clusters, n_features)
            labels (numpy.ndarray): shape (n_samples,)
    """
    # Randomly initialize centroids using random points from the dataset
    centroids = features[np.random.choice(features.shape[0], n_clusters, replace=False)]
    prev_centroids = np.zeros_like(centroids)

    labels = np.zeros(features.shape[0], dtype=int)
    iteration = 0

    while np.linalg.norm(centroids - prev_centroids) > tolerance and iteration < max_iter:
        # E-step: assign labels by nearest centroid
        distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # M-step: update centroids
        prev_centroids = centroids.copy()
        for i in range(n_clusters):
            cluster_points = features[labels == i]
            if len(cluster_points) > 0:  # avoid empty clusters
                centroids[i] = np.mean(cluster_points, axis=0)

        iteration += 1

    return centroids, labels


# Calculate log-likelihood of a clustering (GMM-like evaluation per cluster)
def calculate_log_likelihood(features, centroids, labels, n_clusters):
    """Calculate the log-likelihood of a clustering based on a Gaussian model.

    Args:
        features (numpy.ndarray): shape (n_samples, n_features)
        centroids (numpy.ndarray): shape (n_clusters, n_features)
        labels (numpy.ndarray): shape (n_samples,)
        n_clusters (int): number of clusters

    Returns:
        float: total log-likelihood
    """
    log_likelihood = 0.0
    d = features.shape[1]

    for i in range(n_clusters):
        cluster_points = features[labels == i]
        if len(cluster_points) == 0:
            continue

        # Covariance of points in cluster (+ small regularization)
        cov_matrix = np.cov(cluster_points.T) + np.eye(d) * 1e-6
        mean = centroids[i]

        try:
            cov_inv = np.linalg.inv(cov_matrix)
            cov_det = np.linalg.det(cov_matrix)
        except np.linalg.LinAlgError:
            continue

        # If det is non-positive, skip to avoid invalid sqrt/log
        if cov_det <= 0:
            continue

        norm_const = np.sqrt(((2 * np.pi) ** d) * cov_det)

        for point in cluster_points:
            diff = point - mean
            exponent = -0.5 * (diff @ cov_inv @ diff.T)
            likelihood = np.exp(exponent) / norm_const

            if likelihood > 0:
                log_likelihood += np.log(likelihood)

    return log_likelihood
