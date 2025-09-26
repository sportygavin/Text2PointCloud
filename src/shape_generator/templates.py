"""
Text2PointCloud - Template-Based Point Cloud System

This module handles complex object generation using pre-defined templates.
"""

import numpy as np
from typing import List, Tuple, Dict
import json
import os

class PointCloudTemplates:
    """
    Manages point cloud templates for complex objects.
    """
    
    def __init__(self):
        """Initialize the template system."""
        self.templates = {}
        self.load_basic_templates()
    
    def load_basic_templates(self):
        """Load basic point cloud templates."""
        # Cat template - simplified cat silhouette
        self.templates['cat'] = self._generate_cat_template()
        
        # Dog template - simplified dog silhouette  
        self.templates['dog'] = self._generate_dog_template()
        
        # House template - composite of basic shapes
        self.templates['house'] = self._generate_house_template()
    
    def _generate_cat_template(self) -> List[Tuple[float, float]]:
        """
        Generate a simple cat silhouette point cloud.
        
        Returns:
            List of (x, y) coordinate tuples forming a cat shape
        """
        points = []
        
        # Cat head (circle-like)
        head_points = self._generate_ellipse(0, 0.3, 0.4, 0.3, 50)
        points.extend(head_points)
        
        # Cat ears (triangles)
        left_ear = self._generate_triangle(-0.15, 0.5, 0.1, 0.1, 20)
        right_ear = self._generate_triangle(0.15, 0.5, 0.1, 0.1, 20)
        points.extend(left_ear)
        points.extend(right_ear)
        
        # Cat body (ellipse)
        body_points = self._generate_ellipse(0, -0.2, 0.6, 0.4, 60)
        points.extend(body_points)
        
        # Cat tail (curved line)
        tail_points = self._generate_curved_line(0.3, -0.1, 0.5, 0.3, 30)
        points.extend(tail_points)
        
        return points
    
    def _generate_dog_template(self) -> List[Tuple[float, float]]:
        """
        Generate a simple dog silhouette point cloud.
        
        Returns:
            List of (x, y) coordinate tuples forming a dog shape
        """
        points = []
        
        # Dog head (ellipse)
        head_points = self._generate_ellipse(0, 0.2, 0.5, 0.4, 50)
        points.extend(head_points)
        
        # Dog snout (small ellipse)
        snout_points = self._generate_ellipse(0, 0.1, 0.2, 0.15, 20)
        points.extend(snout_points)
        
        # Dog ears (floppy)
        left_ear = self._generate_ellipse(-0.2, 0.3, 0.15, 0.3, 25)
        right_ear = self._generate_ellipse(0.2, 0.3, 0.15, 0.3, 25)
        points.extend(left_ear)
        points.extend(right_ear)
        
        # Dog body (ellipse)
        body_points = self._generate_ellipse(0, -0.3, 0.7, 0.5, 60)
        points.extend(body_points)
        
        # Dog legs (rectangles)
        for x_offset in [-0.2, -0.1, 0.1, 0.2]:
            leg_points = self._generate_rectangle(x_offset, -0.6, 0.1, 0.3, 20)
            points.extend(leg_points)
        
        return points
    
    def _generate_house_template(self) -> List[Tuple[float, float]]:
        """
        Generate a simple house point cloud.
        
        Returns:
            List of (x, y) coordinate tuples forming a house shape
        """
        points = []
        
        # House base (square)
        base_points = self._generate_rectangle(0, -0.2, 0.8, 0.6, 80)
        points.extend(base_points)
        
        # House roof (triangle)
        roof_points = self._generate_triangle(0, 0.4, 0.8, 0.4, 40)
        points.extend(roof_points)
        
        # Door (rectangle)
        door_points = self._generate_rectangle(0, -0.4, 0.2, 0.3, 20)
        points.extend(door_points)
        
        # Windows (squares)
        left_window = self._generate_rectangle(-0.25, 0.1, 0.15, 0.15, 15)
        right_window = self._generate_rectangle(0.25, 0.1, 0.15, 0.15, 15)
        points.extend(left_window)
        points.extend(right_window)
        
        return points
    
    def _generate_ellipse(self, center_x: float, center_y: float, 
                        width: float, height: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points forming an ellipse."""
        angles = np.linspace(0, 2 * np.pi, num_points)
        x_coords = center_x + (width/2) * np.cos(angles)
        y_coords = center_y + (height/2) * np.sin(angles)
        return list(zip(x_coords, y_coords))
    
    def _generate_triangle(self, center_x: float, center_y: float, 
                          width: float, height: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points forming a triangle."""
        points_per_side = num_points // 3
        
        # Triangle vertices
        top_x = center_x
        top_y = center_y + height/2
        left_x = center_x - width/2
        left_y = center_y - height/2
        right_x = center_x + width/2
        right_y = center_y - height/2
        
        # Generate points along each edge
        x_left = np.linspace(left_x, top_x, points_per_side)
        y_left = np.linspace(left_y, top_y, points_per_side)
        
        x_right = np.linspace(top_x, right_x, points_per_side)
        y_right = np.linspace(top_y, right_y, points_per_side)
        
        x_bottom = np.linspace(right_x, left_x, points_per_side)
        y_bottom = np.linspace(right_y, left_y, points_per_side)
        
        return (list(zip(x_left, y_left)) + 
                list(zip(x_right, y_right)) + 
                list(zip(x_bottom, y_bottom)))
    
    def _generate_rectangle(self, center_x: float, center_y: float, 
                           width: float, height: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points forming a rectangle."""
        points_per_side = num_points // 4
        
        # Rectangle corners
        left = center_x - width/2
        right = center_x + width/2
        top = center_y + height/2
        bottom = center_y - height/2
        
        # Generate points along each edge
        x_top = np.linspace(left, right, points_per_side)
        y_top = np.linspace(top, top, points_per_side)
        
        x_right = np.linspace(right, right, points_per_side)
        y_right = np.linspace(top, bottom, points_per_side)
        
        x_bottom = np.linspace(right, left, points_per_side)
        y_bottom = np.linspace(bottom, bottom, points_per_side)
        
        x_left = np.linspace(left, left, points_per_side)
        y_left = np.linspace(bottom, top, points_per_side)
        
        return (list(zip(x_top, y_top)) + 
                list(zip(x_right, y_right)) + 
                list(zip(x_bottom, y_bottom)) + 
                list(zip(x_left, y_left)))
    
    def _generate_curved_line(self, start_x: float, start_y: float, 
                            end_x: float, end_y: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points forming a curved line."""
        # Simple quadratic curve
        t = np.linspace(0, 1, num_points)
        control_x = (start_x + end_x) / 2
        control_y = max(start_y, end_y) + 0.2  # Curve upward
        
        # Quadratic Bezier curve
        x_coords = (1-t)**2 * start_x + 2*(1-t)*t * control_x + t**2 * end_x
        y_coords = (1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y
        
        return list(zip(x_coords, y_coords))
    
    def get_template(self, object_name: str) -> List[Tuple[float, float]]:
        """
        Get a point cloud template for the specified object.
        
        Args:
            object_name: Name of the object (e.g., 'cat', 'dog', 'house')
            
        Returns:
            List of (x, y) coordinate tuples
        """
        return self.templates.get(object_name.lower(), [])
    
    def list_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
