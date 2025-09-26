"""
Text2PointCloud - Shape Generation Module

This module generates point clouds for various shapes and concepts.
"""

import numpy as np
from typing import List, Tuple

class ShapeGenerator:
    """
    Generates point clouds for different shapes and concepts.
    """
    
    def __init__(self):
        """Initialize the shape generator."""
        pass
    
    def generate_circle(self, radius: float = 1.0, num_points: int = 100) -> List[Tuple[float, float]]:
        """
        Generate points that form a circle.
        
        Args:
            radius: Radius of the circle
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) coordinate tuples
        """

        angles = np.linspace(0, 2 * np.pi, num_points)

        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        return list(zip(x_coords, y_coords))
    
    def generate_square(self, size: float = 1.0, num_points: int = 100) -> List[Tuple[float, float]]:
        """
        Generate points that form a square.
        
        Args:
            size: Size of the square
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) coordinate tuples
        """

        # TODO: Implement square generation
        corners = {
            'top_left': (-size/2, size/2),
            'top_right': (size/2, size/2),
            'bottom_right': (size/2, -size/2),
            'bottom_left': (-size/2, -size/2)
        }

        points = []
        points_per_side = num_points // 4

        # Top edge: x goes from -size/2 to size/2, y stays at size/2
        x_top = np.linspace(-size/2, size/2, points_per_side)
        y_top = np.linspace(size/2, size/2, points_per_side) 

        # Right edge: y goes from size/2 to -size/2, x stays at size/2  
        x_right = np.linspace(size/2, size/2, points_per_side) 
        y_right = np.linspace(size/2, -size/2, points_per_side)

        # Bottom edge: x goes from -size/2 to size/2, y stays at size/2
        x_bottom = np.linspace(-size/2, size/2, points_per_side)
        y_bottom = np.linspace(-size/2, -size/2, points_per_side) 

        # Left edge: y goes from size/2 to -size/2, x stays at size/2  
        x_left = np.linspace(-size/2, -size/2, points_per_side) 
        y_left = np.linspace(-size/2, size/2, points_per_side)

        return (list(zip(x_top, y_top)) + list(zip(x_right, y_right)) + list(zip(x_bottom, y_bottom)) + list(zip(x_left, y_left)))

        
    
    def generate_triangle(self, size: float = 1.0, num_points: int = 100) -> List[Tuple[float, float]]:
        """
        Generate points that form a triangle.
        
        Args:
            size: Size of the triangle
            num_points: Number of points to generate
            
        Returns:
            List of (x, y) coordinate tuples
        """
        # Calculate points per side
        points_per_side = num_points // 3
        
        # Define triangle corners (equilateral triangle)
        # Top vertex
        top_x = 0
        top_y = size * 0.866  # height of equilateral triangle
        
        # Bottom left vertex  
        bottom_left_x = -size/2
        bottom_left_y = -size * 0.289  # height from center to base
        
        # Bottom right vertex
        bottom_right_x = size/2
        bottom_right_y = -size * 0.289
        
        # Generate points along each edge
        
        # Left edge: from bottom_left to top
        x_left = np.linspace(bottom_left_x, top_x, points_per_side)
        y_left = np.linspace(bottom_left_y, top_y, points_per_side)
        
        # Right edge: from top to bottom_right
        x_right = np.linspace(top_x, bottom_right_x, points_per_side)
        y_right = np.linspace(top_y, bottom_right_y, points_per_side)
        
        # Bottom edge: from bottom_right to bottom_left
        x_bottom = np.linspace(bottom_right_x, bottom_left_x, points_per_side)
        y_bottom = np.linspace(bottom_right_y, bottom_left_y, points_per_side)
        
        return (list(zip(x_left, y_left)) + 
                list(zip(x_right, y_right)) + 
                list(zip(x_bottom, y_bottom)))
