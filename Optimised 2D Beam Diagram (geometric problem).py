# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:15:52 2025

@author: Hamish
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Number of reflections
n_reflections = 8

# Define beam path y-position
beam_y = 50

# Create non-uniform blue line positions (frame numbers)
x_blues_ideal = np.sort(np.random.uniform(10, 90, n_reflections))

# Generate white Bragg lines so they intersect blue lines near (but not exactly on) the red line
small_offsets = np.random.uniform(-5, 5, n_reflections)
slopes = np.random.uniform(-3, 3, n_reflections)
intercepts = beam_y + small_offsets - slopes * x_blues_ideal

# Apply a random horizontal shift to simulate experimental error (at least 10 units)
random_shift = np.random.uniform(10, 20) * np.random.choice([-1, 1])
x_blues_shifted = x_blues_ideal + random_shift

# Function to compute yellow dots for any horizontal shift
def compute_yellow_dots(x_shift):
    intersections = []
    for i in range(n_reflections):
        x = x_blues_shifted[i] + x_shift
        y = slopes[i] * x + intercepts[i]
        intersections.append((x, y))
    return np.array(intersections)

# Objective: minimise vertical distance of yellow dots from red beam path
def objective(x_shift):
    yellow_dots = compute_yellow_dots(x_shift)
    deviations = yellow_dots[:, 1] - beam_y
    return np.sum(deviations**2)

# Optimise the shift
result = minimize_scalar(objective)
optimal_shift = result.x
yellow_dots_initial = compute_yellow_dots(0)
yellow_dots_optimized = compute_yellow_dots(optimal_shift)

# Calculate number of frames shifted using scale: 2 units = 1 frame
frames_shifted = optimal_shift / 2

# Function to draw yellow line centered on dot, angled like the white line,
# with horizontal projection of ±2 units (i.e. ±1 frame)
def draw_yellow_segment(ax, x0, y0, slope):
    dx = 2  # ±2 units horizontally
    dy = slope * dx
    x_start = x0 - dx
    y_start = y0 - dy
    x_end = x0 + dx
    y_end = y0 + dy
    ax.plot([x_start, x_end], [y_start, y_end], color='yellow', linewidth=2, zorder=10)

# --- Plotting both arrangements ---

fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
x_vals = np.linspace(0, 100, 500)

# Plot 1: Initial misaligned setup
ax[0].set_title("Simulation output without optimisation")
for i in range(n_reflections):
    y_vals = slopes[i] * x_vals + intercepts[i]
    ax[0].plot(x_vals, y_vals, color='white', linewidth=1)
    ax[0].axvline(x_blues_shifted[i], color='blue')
    x_dot, y_dot = yellow_dots_initial[i]
    ax[0].plot(x_dot, y_dot, 'yo', zorder=10)
    draw_yellow_segment(ax[0], x_dot, y_dot, slopes[i])
ax[0].axhline(beam_y, color='red', linestyle='-', label='Beam path')
ax[0].set_facecolor('black')
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 100)
ax[0].set_xticks(np.arange(0, 101, 10))
ax[0].set_xlabel("Frame number")

# Plot 2: Optimised alignment
ax[1].set_title(f"After optimisation: horizontal shift = {frames_shifted:.2f} frames")
for i in range(n_reflections):
    y_vals = slopes[i] * x_vals + intercepts[i]
    ax[1].plot(x_vals, y_vals, color='white', linewidth=1)
    ax[1].axvline(x_blues_shifted[i] + optimal_shift, color='blue')
    x_dot, y_dot = yellow_dots_optimized[i]
    ax[1].plot(x_dot, y_dot, 'yo', zorder=10)
    draw_yellow_segment(ax[1], x_dot, y_dot, slopes[i])
ax[1].axhline(beam_y, color='red', linestyle='-', label='Beam path')
ax[1].set_facecolor('black')
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
ax[1].set_xticks(np.arange(0, 101, 10))
ax[1].set_xlabel("Frame number")

plt.tight_layout()
plt.show()

# Output optimal shift
print(f"Optimal horizontal shift to align yellow dots: {optimal_shift:.2f} units")
print(f"Which corresponds to: {frames_shifted:.2f} frames")

