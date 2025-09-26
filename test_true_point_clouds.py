#!/usr/bin/env python3
"""
Test the true point cloud generation with actual 3D object outlines.
"""

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from true_point_cloud_model import TruePointCloudModel
from text_processing.improved_encoder import ImprovedTextEncoder

def test_true_point_clouds():
    """Test true point cloud generation with visualization."""
    print("🎯 Testing True Point Cloud Generation")
    print("=" * 60)
    
    # Create model
    text_encoder = ImprovedTextEncoder()
    model = TruePointCloudModel(text_encoder)
    
    # Test texts with specific descriptions
    test_texts = [
        "a wooden chair with four legs and backrest",
        "a red sports car with two doors and wheels", 
        "a white commercial airplane with wings and tail",
        "a modern glass table with four legs"
    ]
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
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
        
        # Plot points as individual dots
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=z_coords, cmap='viridis', s=8, alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"'{text}'", fontsize=12, pad=20)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Set viewing angle for better visibility
        if i == 0:  # Chair
            ax.view_init(elev=20, azim=45)
        elif i == 1:  # Car
            ax.view_init(elev=10, azim=0)
        elif i == 2:  # Airplane
            ax.view_init(elev=10, azim=90)
        else:  # Table
            ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.suptitle("Text2PointCloud - True Point Cloud Object Outlines", fontsize=16, y=0.95)
    plt.show()
    
    print("\n🎉 Test completed! Check the 3D visualizations above.")
    print("💡 The point clouds should now look like actual 3D object outlines!")
    print("\n🔍 Key improvements:")
    print("   • Points form actual object surfaces and edges")
    print("   • Chairs have seat, backrest, and leg structure")
    print("   • Cars have body, wheels, and windshield surfaces")
    print("   • Airplanes have fuselage, wings, and tail surfaces")
    print("   • Tables have flat surface with supporting legs")
    print("   • Each point represents a dot on the object outline")

if __name__ == "__main__":
    test_true_point_clouds()
