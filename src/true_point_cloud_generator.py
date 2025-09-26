"""
Text2PointCloud - True Point Cloud Outline Generator

This module generates actual point cloud outlines that look like 3D objects
using dots that form the surface and edges of objects.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

class TruePointCloudGenerator:
    """
    Generates actual point cloud outlines that look like 3D objects.
    """
    
    def __init__(self):
        """Initialize the point cloud generator."""
        self.object_templates = {
            'chair': self._generate_chair_point_cloud,
            'table': self._generate_table_point_cloud,
            'car': self._generate_car_point_cloud,
            'airplane': self._generate_airplane_point_cloud,
            'lamp': self._generate_lamp_point_cloud,
            'sofa': self._generate_sofa_point_cloud,
            'bed': self._generate_bed_point_cloud,
            'desk': self._generate_desk_point_cloud,
            'bookcase': self._generate_bookcase_point_cloud,
            'bicycle': self._generate_bicycle_point_cloud
        }
    
    def generate_point_cloud(self, text: str, num_points: int = 1024) -> List[Tuple[float, float, float]]:
        """
        Generate point cloud outline from text description.
        
        Args:
            text: Text description of the object
            num_points: Number of points to generate
            
        Returns:
            List of (x, y, z) coordinate tuples forming object outline
        """
        # Extract object type from text
        object_type = self._extract_object_type(text)
        
        # Generate point cloud based on object type
        if object_type in self.object_templates:
            points = self.object_templates[object_type](text, num_points)
        else:
            # Fallback to generic point cloud
            points = self._generate_generic_point_cloud(text, num_points)
        
        return points
    
    def _extract_object_type(self, text: str) -> str:
        """Extract object type from text description."""
        text_lower = text.lower()
        
        # Check for specific object types
        for obj_type in self.object_templates.keys():
            if obj_type in text_lower:
                return obj_type
        
        # Check for synonyms
        synonyms = {
            'seat': 'chair', 'furniture': 'chair', 'stool': 'chair',
            'surface': 'table', 'desk': 'table', 'counter': 'table',
            'vehicle': 'car', 'automobile': 'car', 'truck': 'car',
            'plane': 'airplane', 'aircraft': 'airplane', 'jet': 'airplane',
            'light': 'lamp', 'lighting': 'lamp',
            'couch': 'sofa', 'settee': 'sofa',
            'mattress': 'bed',
            'workstation': 'desk',
            'shelf': 'bookcase',
            'bike': 'bicycle', 'cycle': 'bicycle'
        }
        
        for synonym, obj_type in synonyms.items():
            if synonym in text_lower:
                return obj_type
        
        return 'generic'
    
    def _generate_chair_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate chair point cloud that looks like an actual chair."""
        points = []
        
        # Seat surface (rectangular grid of points)
        seat_points = self._generate_surface_points(
            center=(0, 0, 0.1), width=0.5, height=0.5, 
            num_points=num_points // 6
        )
        points.extend(seat_points)
        
        # Backrest (vertical surface)
        backrest_points = self._generate_surface_points(
            center=(0, 0.25, 0.4), width=0.5, height=0.6,
            num_points=num_points // 6
        )
        points.extend(backrest_points)
        
        # Legs (4 vertical lines of points)
        leg_positions = [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_points(
                start=(x, y, -0.3), end=(x, y, 0.1),
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        # Armrests (if mentioned)
        if 'arm' in text.lower() or 'armrest' in text.lower():
            for side in [-0.25, 0.25]:
                # Armrest surface
                armrest_points = self._generate_surface_points(
                    center=(side, 0, 0.3), width=0.1, height=0.4,
                    num_points=num_points // 20
                )
                points.extend(armrest_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_table_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate table point cloud that looks like an actual table."""
        points = []
        
        # Tabletop (rectangular surface)
        top_points = self._generate_surface_points(
            center=(0, 0, 0.4), width=0.8, height=0.8,
            num_points=num_points // 4
        )
        points.extend(top_points)
        
        # Legs (4 vertical lines of points)
        leg_positions = [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)]
        for x, y in leg_positions:
            leg_points = self._generate_line_points(
                start=(x, y, -0.4), end=(x, y, 0.4),
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        # Round table variation
        if 'round' in text.lower() or 'circular' in text.lower():
            # Make tabletop circular
            for i in range(len(top_points)):
                x, y, z = top_points[i]
                if z > 0.3:  # Tabletop
                    angle = np.arctan2(y, x)
                    radius = 0.4
                    top_points[i] = (radius * np.cos(angle), radius * np.sin(angle), z)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_car_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate car point cloud that looks like an actual car."""
        points = []
        
        # Car body (ellipsoid surface)
        body_points = self._generate_ellipsoid_surface(
            center=(0, 0, 0), radii=(0.4, 0.2, 0.15),
            num_points=num_points // 3
        )
        points.extend(body_points)
        
        # Wheels (4 circular surfaces)
        wheel_positions = [(-0.3, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.3, 0.2)]
        for x, y in wheel_positions:
            wheel_points = self._generate_circle_surface(
                center=(x, y, -0.2), radius=0.1,
                num_points=num_points // 20
            )
            points.extend(wheel_points)
        
        # Windshield (front surface)
        windshield_points = self._generate_surface_points(
            center=(0.1, 0, 0.05), width=0.3, height=0.2,
            num_points=num_points // 15
        )
        points.extend(windshield_points)
        
        # Sports car variation
        if 'sports' in text.lower() or 'sleek' in text.lower():
            # Lower and longer profile
            for i in range(len(points)):
                x, y, z = points[i]
                if z > 0:  # Upper body
                    points[i] = (x * 1.2, y, z * 0.7)  # Longer and lower
                else:  # Lower body
                    points[i] = (x * 1.2, y, z)  # Just longer
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_airplane_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate airplane point cloud that looks like an actual airplane."""
        points = []
        
        # Fuselage (elongated ellipsoid)
        fuselage_points = self._generate_ellipsoid_surface(
            center=(0, 0, 0), radii=(0.5, 0.1, 0.1),
            num_points=num_points // 4
        )
        points.extend(fuselage_points)
        
        # Wings (horizontal surfaces)
        wing_points = self._generate_surface_points(
            center=(0, 0, 0), width=0.2, height=0.8,
            num_points=num_points // 6
        )
        points.extend(wing_points)
        
        # Tail (vertical surface)
        tail_points = self._generate_surface_points(
            center=(-0.4, 0, 0.2), width=0.1, height=0.3,
            num_points=num_points // 10
        )
        points.extend(tail_points)
        
        # Nose (pointed front)
        nose_points = self._generate_line_points(
            start=(0.4, 0, 0), end=(0.5, 0, 0),
            num_points=num_points // 20
        )
        points.extend(nose_points)
        
        # Commercial airliner variation
        if 'commercial' in text.lower() or 'large' in text.lower():
            # Scale up
            for i in range(len(points)):
                x, y, z = points[i]
                points[i] = (x * 1.3, y * 1.3, z * 1.3)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_lamp_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate lamp point cloud that looks like an actual lamp."""
        points = []
        
        # Base (circular surface)
        base_points = self._generate_circle_surface(
            center=(0, 0, -0.1), radius=0.2,
            num_points=num_points // 8
        )
        points.extend(base_points)
        
        # Pole (vertical line of points)
        pole_points = self._generate_line_points(
            start=(0, 0, -0.1), end=(0, 0, 0.8),
            num_points=num_points // 4
        )
        points.extend(pole_points)
        
        # Shade (inverted cone surface)
        shade_points = self._generate_cone_surface(
            center=(0, 0, 0.8), radius=0.3, height=0.2,
            num_points=num_points // 4
        )
        points.extend(shade_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_sofa_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate sofa point cloud that looks like an actual sofa."""
        points = []
        
        # Seat (rectangular surface)
        seat_points = self._generate_surface_points(
            center=(0, 0, 0.1), width=1.2, height=0.6,
            num_points=num_points // 6
        )
        points.extend(seat_points)
        
        # Backrest (vertical surface)
        backrest_points = self._generate_surface_points(
            center=(0, 0.3, 0.4), width=1.2, height=0.6,
            num_points=num_points // 6
        )
        points.extend(backrest_points)
        
        # Armrests
        for side in [-0.6, 0.6]:
            armrest_points = self._generate_surface_points(
                center=(side, 0, 0.25), width=0.1, height=0.6,
                num_points=num_points // 12
            )
            points.extend(armrest_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.5, -0.2), (0.5, -0.2), (-0.5, 0.2), (0.5, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_points(
                start=(x, y, -0.3), end=(x, y, 0.1),
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bed_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bed point cloud that looks like an actual bed."""
        points = []
        
        # Mattress (rectangular surface)
        mattress_points = self._generate_surface_points(
            center=(0, 0, 0.2), width=1.6, height=0.8,
            num_points=num_points // 4
        )
        points.extend(mattress_points)
        
        # Headboard (vertical surface)
        headboard_points = self._generate_surface_points(
            center=(0, 0.4, 0.6), width=1.6, height=0.8,
            num_points=num_points // 8
        )
        points.extend(headboard_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3)]
        for x, y in leg_positions:
            leg_points = self._generate_line_points(
                start=(x, y, -0.2), end=(x, y, 0.2),
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_desk_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate desk point cloud that looks like an actual desk."""
        points = []
        
        # Desktop (rectangular surface)
        desktop_points = self._generate_surface_points(
            center=(0, 0, 0.3), width=1.0, height=0.6,
            num_points=num_points // 4
        )
        points.extend(desktop_points)
        
        # Drawers (rectangular box)
        drawer_points = self._generate_surface_points(
            center=(0, -0.3, 0.15), width=0.8, height=0.2,
            num_points=num_points // 8
        )
        points.extend(drawer_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.4, -0.2), (0.4, -0.2), (-0.4, 0.2), (0.4, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_points(
                start=(x, y, -0.2), end=(x, y, 0.3),
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bookcase_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bookcase point cloud that looks like an actual bookcase."""
        points = []
        
        # Main structure (vertical surfaces)
        structure_points = self._generate_surface_points(
            center=(0, 0, 0.8), width=0.4, height=1.6,
            num_points=num_points // 4
        )
        points.extend(structure_points)
        
        # Shelves (horizontal surfaces)
        shelf_heights = [-0.4, 0, 0.4, 0.8, 1.2]
        for height in shelf_heights:
            shelf_points = self._generate_surface_points(
                center=(0, 0, height), width=0.4, height=1.6,
                num_points=num_points // 20
            )
            points.extend(shelf_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bicycle_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bicycle point cloud that looks like an actual bicycle."""
        points = []
        
        # Frame (main structure)
        frame_points = self._generate_line_points(
            start=(-0.3, 0, 0.3), end=(0.3, 0, 0.3),
            num_points=num_points // 8
        )
        points.extend(frame_points)
        
        # Wheels (2 circular surfaces)
        wheel_positions = [(-0.3, 0), (0.3, 0)]
        for x, y in wheel_positions:
            wheel_points = self._generate_circle_surface(
                center=(x, y, 0), radius=0.2,
                num_points=num_points // 8
            )
            points.extend(wheel_points)
        
        # Handlebars
        handlebar_points = self._generate_line_points(
            start=(-0.3, 0, 0.3), end=(-0.3, 0, 0.5),
            num_points=num_points // 16
        )
        points.extend(handlebar_points)
        
        # Seat
        seat_points = self._generate_circle_surface(
            center=(0, 0, 0.3), radius=0.05,
            num_points=num_points // 16
        )
        points.extend(seat_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_generic_point_cloud(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate generic point cloud for unknown objects."""
        points = []
        
        # Main body (ellipsoid surface)
        body_points = self._generate_ellipsoid_surface(
            center=(0, 0, 0), radii=(0.3, 0.3, 0.3),
            num_points=num_points // 2
        )
        points.extend(body_points)
        
        # Add some structure
        structure_points = self._generate_surface_points(
            center=(0, 0, 0), width=0.4, height=0.4,
            num_points=num_points // 4
        )
        points.extend(structure_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_surface_points(self, center: Tuple[float, float, float], 
                               width: float, height: float, 
                               num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points on a rectangular surface."""
        points = []
        cx, cy, cz = center
        
        # Generate grid of points on the surface
        for i in range(num_points):
            # Random points on the surface
            x = cx + np.random.uniform(-width/2, width/2)
            y = cy + np.random.uniform(-height/2, height/2)
            z = cz
            points.append((x, y, z))
        
        return points
    
    def _generate_line_points(self, start: Tuple[float, float, float], 
                            end: Tuple[float, float, float], 
                            num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points along a line."""
        points = []
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            points.append((x, y, z))
        
        return points
    
    def _generate_circle_surface(self, center: Tuple[float, float, float], 
                               radius: float, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points on a circular surface."""
        points = []
        cx, cy, cz = center
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            z = cz
            points.append((x, y, z))
        
        return points
    
    def _generate_ellipsoid_surface(self, center: Tuple[float, float, float], 
                                  radii: Tuple[float, float, float], 
                                  num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points on an ellipsoid surface."""
        points = []
        cx, cy, cz = center
        rx, ry, rz = radii
        
        for i in range(num_points):
            # Use spherical coordinates
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, np.pi)
            
            x = cx + rx * np.sin(v) * np.cos(u)
            y = cy + ry * np.sin(v) * np.sin(u)
            z = cz + rz * np.cos(v)
            
            points.append((x, y, z))
        
        return points
    
    def _generate_cone_surface(self, center: Tuple[float, float, float], 
                             radius: float, height: float, 
                             num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points on a cone surface."""
        points = []
        cx, cy, cz = center
        
        for i in range(num_points):
            # Random points on cone surface
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius)
            h = np.random.uniform(0, height)
            
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            z = cz + h
            
            points.append((x, y, z))
        
        return points

# Test function
def test_true_point_cloud_generator():
    """Test the true point cloud generator."""
    generator = TruePointCloudGenerator()
    
    test_texts = [
        "a wooden chair with four legs",
        "a red sports car with two doors",
        "a white commercial airplane",
        "a modern glass table"
    ]
    
    for text in test_texts:
        print(f"\nGenerating point cloud for: '{text}'")
        points = generator.generate_point_cloud(text, 200)
        print(f"Generated {len(points)} points")
        print(f"First 5 points: {points[:5]}")

if __name__ == "__main__":
    test_true_point_cloud_generator()
