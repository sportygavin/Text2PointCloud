"""
Text2PointCloud - 3D Visualization Module

This module handles advanced 3D visualization of point clouds.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Tuple
import ipywidgets as widgets
from IPython.display import display

class PointCloudVisualizer3D:
    """
    Advanced 3D point cloud visualization with multiple rendering options.
    """
    
    def __init__(self):
        """Initialize the 3D visualizer."""
        pass
    
    def plot_matplotlib_3d(self, points: List[Tuple[float, float, float]], 
                          title: str = "3D Point Cloud", 
                          color_by: str = 'z',
                          size: int = 20,
                          alpha: float = 0.7):
        """
        Plot 3D points using matplotlib with advanced features.
        
        Args:
            points: List of (x, y, z) coordinate tuples
            title: Title for the plot
            color_by: Color points by 'z', 'distance', or 'random'
            size: Size of points
            alpha: Transparency of points
        """
        if not points:
            print(f"No points to plot for {title}")
            return
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Main 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Color points based on criteria
        if color_by == 'z':
            colors = z_coords
            cmap = 'viridis'
        elif color_by == 'distance':
            distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_coords, y_coords, z_coords)]
            colors = distances
            cmap = 'plasma'
        else:  # random
            colors = np.random.rand(len(points))
            cmap = 'tab10'
        
        scatter = ax1.scatter(x_coords, y_coords, z_coords, 
                             c=colors, s=size, alpha=alpha, cmap=cmap)
        ax1.set_title(f"{title}\n(3D View)")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # XY projection
        ax2 = fig.add_subplot(132)
        ax2.scatter(x_coords, y_coords, c=colors, s=size, alpha=alpha, cmap=cmap)
        ax2.set_title("XY Projection")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        # XZ projection
        ax3 = fig.add_subplot(133)
        ax3.scatter(x_coords, z_coords, c=colors, s=size, alpha=alpha, cmap=cmap)
        ax3.set_title("XZ Projection")
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_plotly_3d(self, points: List[Tuple[float, float, float]], 
                       title: str = "3D Point Cloud",
                       color_by: str = 'z',
                       size: int = 3):
        """
        Plot 3D points using plotly for interactive visualization.
        
        Args:
            points: List of (x, y, z) coordinate tuples
            title: Title for the plot
            color_by: Color points by 'z', 'distance', or 'random'
            size: Size of points
        """
        if not points:
            print(f"No points to plot for {title}")
            return
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Color points based on criteria
        if color_by == 'z':
            colors = z_coords
            colorbar_title = "Z Coordinate"
        elif color_by == 'distance':
            distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_coords, y_coords, z_coords)]
            colors = distances
            colorbar_title = "Distance from Origin"
        else:  # random
            colors = np.random.rand(len(points))
            colorbar_title = "Random"
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=colorbar_title)
            ),
            text=[f"Point {i}" for i in range(len(points))],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    
    def plot_multiple_views(self, points: List[Tuple[float, float, float]], 
                           title: str = "3D Point Cloud"):
        """
        Plot multiple views of the same point cloud.
        """
        if not points:
            print(f"No points to plot for {title}")
            return
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Create subplots
        fig = plt.figure(figsize=(20, 5))
        
        views = [
            (ax1 := fig.add_subplot(141, projection='3d'), "3D View"),
            (ax2 := fig.add_subplot(142), "XY View"),
            (ax3 := fig.add_subplot(143), "XZ View"),
            (ax4 := fig.add_subplot(144), "YZ View")
        ]
        
        for i, (ax, view_title) in enumerate(views):
            if i == 0:  # 3D view
                ax.scatter(x_coords, y_coords, z_coords, s=20, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            elif i == 1:  # XY view
                ax.scatter(x_coords, y_coords, s=20, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
            elif i == 2:  # XZ view
                ax.scatter(x_coords, z_coords, s=20, alpha=0.7)
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
            else:  # YZ view
                ax.scatter(y_coords, z_coords, s=20, alpha=0.7)
                ax.set_xlabel('Y')
                ax.set_ylabel('Z')
            
            ax.set_title(view_title)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def analyze_point_cloud(self, points: List[Tuple[float, float, float]]):
        """
        Analyze and display statistics about the point cloud.
        
        Args:
            points: List of (x, y, z) coordinate tuples
        """
        if not points:
            print("No points to analyze")
            return
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Calculate statistics
        stats = {
            'Total Points': len(points),
            'X Range': (min(x_coords), max(x_coords)),
            'Y Range': (min(y_coords), max(y_coords)),
            'Z Range': (min(z_coords), max(z_coords)),
            'Center': (np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)),
            'Std Dev': (np.std(x_coords), np.std(y_coords), np.std(z_coords))
        }
        
        print(f"Point Cloud Analysis:")
        print(f"===================")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats

# Example usage
def test_3d_visualization():
    """Test the 3D visualization capabilities."""
    print("Testing 3D Point Cloud Visualization...")
    
    # Generate some test points
    np.random.seed(42)
    test_points = [(np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)) 
                   for _ in range(100)]
    
    # Create visualizer
    visualizer = PointCloudVisualizer3D()
    
    # Test different visualization methods
    print("\n1. Matplotlib 3D with projections...")
    visualizer.plot_matplotlib_3d(test_points, "Test Point Cloud", color_by='z')
    
    print("\n2. Plotly interactive 3D...")
    visualizer.plot_plotly_3d(test_points, "Interactive Test Point Cloud", color_by='distance')
    
    print("\n3. Multiple views...")
    visualizer.plot_multiple_views(test_points, "Multiple Views Test")
    
    print("\n4. Point cloud analysis...")
    visualizer.analyze_point_cloud(test_points)

if __name__ == "__main__":
    test_3d_visualization()
