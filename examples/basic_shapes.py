"""
Basic Shape Generation Example

This example demonstrates how to generate simple 2D point clouds.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_circle_points(radius=1.0, num_points=100):
    """
    Generate points that form a circle using parametric equations.
    
    Args:
        radius: Radius of the circle
        num_points: Number of points to generate
        
    Returns:
        List of (x, y) coordinate tuples
    """
    # Generate angles from 0 to 2π
    angles = np.linspace(0, 2 * np.pi, num_points)
    
    # Calculate x and y coordinates using parametric equations
    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)
    
    # Return as list of tuples
    return list(zip(x_coords, y_coords))

def plot_points(points, title="Point Cloud"):
    """
    Plot points using matplotlib.
    
    Args:
        points: List of (x, y) coordinate tuples
        title: Title for the plot
    """
    x_coords, y_coords = zip(*points)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=20, alpha=0.7)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Generate and display a circle
    circle_points = generate_circle_points(radius=2.0, num_points=200)
    plot_points(circle_points, "Circle Point Cloud")
    
    print(f"Generated {len(circle_points)} points for circle")
    print("First 5 points:", circle_points[:5])
