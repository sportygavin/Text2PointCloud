"""
Test script for ShapeGenerator class

This script tests the shape generation functions.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add src directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from shape_generator import ShapeGenerator

def plot_points(points, title="Point Cloud"):
    """
    Plot points using matplotlib.
    
    Args:
        points: List of (x, y) coordinate tuples
        title: Title for the plot
    """
    if not points:
        print(f"No points to plot for {title}")
        return
        
    x_coords, y_coords = zip(*points)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=20, alpha=0.7)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_shape_generator():
    """Test the ShapeGenerator class."""
    print("Testing ShapeGenerator...")
    
    # Create a shape generator instance
    generator = ShapeGenerator()
    
    # Test circle generation
    print("\n1. Testing circle generation...")
    try:
        circle_points = generator.generate_circle(radius=2.0, num_points=100)
        print(f"✓ Circle: Generated {len(circle_points)} points")
        print(f"  First 3 points: {circle_points[:3]}")
        plot_points(circle_points, "Circle Point Cloud")
    except Exception as e:
        print(f"✗ Circle failed: {e}")
    
    # Test square generation
    print("\n2. Testing square generation...")
    try:
        square_points = generator.generate_square(size=2.0, num_points=100)
        print(f"✓ Square: Generated {len(square_points)} points")
        print(f"  First 3 points: {square_points[:3]}")
        plot_points(square_points, "Square Point Cloud")
    except Exception as e:
        print(f"✗ Square failed: {e}")
    
    # Test triangle generation
    print("\n3. Testing triangle generation...")
    try:
        triangle_points = generator.generate_triangle(size=2.0, num_points=100)
        print(f"✓ Triangle: Generated {len(triangle_points)} points")
        print(f"  First 3 points: {triangle_points[:3]}")
        plot_points(triangle_points, "Triangle Point Cloud")
    except Exception as e:
        print(f"✗ Triangle failed: {e}")

if __name__ == "__main__":
    test_shape_generator()
