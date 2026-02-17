#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:36:39 2026

@author: surindersinghchhabra
"""

import pandas as pd
import matplotlib.pyplot as plt


# Load Gaia dataset
data = pd.read_csv("gaia_data.csv")


# ---------------------------
# Sky distribution: RA vs Dec
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(data["ra"], data["dec"], alpha=0.5, s=1)

plt.title("Sky Distribution: RA vs Dec")
plt.xlabel("Right Ascension (RA)")
plt.ylabel("Declination (Dec)")
plt.grid(True)

plt.show()


# ---------------------------
# Distribution of Parallax
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hist(data["parallax"], bins=50, color="skyblue", edgecolor="black")

plt.title("Distribution of Parallax")
plt.xlabel("Parallax (mas)")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()


# ---------------------------
# Distribution of G Magnitude
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hist(data["phot_g_mean_mag"], bins=50, color="orange", edgecolor="black")

plt.title("Distribution of G Magnitude")
plt.xlabel("G Magnitude")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()


# ---------------------------
# Distribution of Star Distances
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hist(data["distance_gspphot"], bins=50, color="green", edgecolor="black")

plt.title("Distribution of Star Distances")
plt.xlabel("Distance (pc)")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()


# ---------------------------
# Density of stars (Hexbin)
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hexbin(data["ra"], data["dec"], gridsize=50, cmap="YlGnBu")

plt.colorbar(label="Density")
plt.title("Density of Stars (RA vs Dec)")
plt.xlabel("Right Ascension (RA)")
plt.ylabel("Declination (Dec)")

plt.show()


# ---------------------------------------------
# RA vs Dec colored by distance
# ---------------------------------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    data["ra"],
    data["dec"],
    c=data["distance_gspphot"],
    cmap="viridis",
    s=40,
    alpha=0.6,
    edgecolors="w",
)

plt.colorbar(scatter, label="D
