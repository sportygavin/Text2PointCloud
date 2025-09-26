"""
Test script for complex point cloud templates

This script tests the template-based point cloud generation.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from shape_generator.templates import PointCloudTemplates

def plot_points(points, title="Point Cloud"):
    """Plot points using matplotlib."""
    if not points:
        print(f"No points to plot for {title}")
        return
        
    x_coords, y_coords = zip(*points)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, s=15, alpha=0.7, c='blue')
    plt.title(title, fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

def test_complex_templates():
    """Test the complex point cloud templates."""
    print("Testing Complex Point Cloud Templates...")
    
    # Create template system
    templates = PointCloudTemplates()
    
    # List available templates
    available = templates.list_available_templates()
    print(f"Available templates: {available}")
    
    # Test each template
    for template_name in available:
        print(f"\nTesting {template_name} template...")
        try:
            points = templates.get_template(template_name)
            print(f"✓ {template_name}: Generated {len(points)} points")
            print(f"  First 3 points: {points[:3]}")
            plot_points(points, f"{template_name.title()} Point Cloud")
        except Exception as e:
            print(f"✗ {template_name} failed: {e}")

if __name__ == "__main__":
    test_complex_templates()
