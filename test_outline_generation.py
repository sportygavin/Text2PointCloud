#!/usr/bin/env python3
"""
Test the outline-based point cloud generation.
"""

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from outline_model import OutlineBasedModel
from text_processing.improved_encoder import ImprovedTextEncoder

def test_outline_generation():
    """Test outline generation with visualization."""
    print("🎯 Testing Outline-Based Point Cloud Generation")
    print("=" * 60)
    
    # Create model
    text_encoder = ImprovedTextEncoder()
    model = OutlineBasedModel(text_encoder)
    
    # Test texts
    test_texts = [
        "a wooden chair with four legs",
        "a red sports car with two doors", 
        "a white commercial airplane",
        "a modern glass table"
    ]
    
    # Create subplots
    fig = plt.figure(figsize=(16, 12))
    
    for i, text in enumerate(test_texts):
        print(f"\n🔄 Generating: '{text}'")
        
        # Generate point cloud
        points = model.generate_point_cloud(text)
        print(f"✅ Generated {len(points)} points")
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        print(f"📊 X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"📊 Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"📊 Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
        
        # Create 3D subplot
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Plot points
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=z_coords, cmap='viridis', s=20, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"'{text}'")
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.suptitle("Text2PointCloud - Object Outline Generation", fontsize=16, y=0.98)
    plt.show()
    
    print("\n🎉 Test completed! Check the 3D visualizations above.")
    print("💡 The point clouds should now look like actual object outlines!")

if __name__ == "__main__":
    test_outline_generation()
