"""
Test script for ML-based point cloud generation

This script tests the machine learning approach to point cloud generation.
"""

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def plot_3d_points(points, title="3D Point Cloud"):
    """Plot 3D points using matplotlib."""
    if not points:
        print(f"No points to plot for {title}")
        return
    
    # Extract x, y, z coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x_coords, y_coords, z_coords, s=20, alpha=0.7)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

def test_ml_generation():
    """Test the ML-based point cloud generation."""
    print("Testing ML-based Point Cloud Generation...")
    
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        
        # Initialize generator
        generator = TextToPointCloudML()
        
        # Test with different text inputs
        test_texts = [
            "a red car",
            "a fluffy cat", 
            "a tall tree",
            "a modern house",
            "a flying bird"
        ]
        
        for text in test_texts:
            print(f"\nGenerating point cloud for: '{text}'")
            try:
                points = generator.generate_point_cloud(text, num_points=100)
                print(f"✓ Generated {len(points)} points")
                print(f"  First 3 points: {points[:3]}")
                
                # Plot the 3D point cloud
                plot_3d_points(points, f"ML Generated: {text}")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PyTorch is installed: pip install torch")

if __name__ == "__main__":
    test_ml_generation()
