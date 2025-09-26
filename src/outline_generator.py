"""
Text2PointCloud - Object Outline Generator

This module generates actual object outlines instead of random blobs.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

class ObjectOutlineGenerator:
    """
    Generates actual object outlines from text descriptions.
    """
    
    def __init__(self):
        """Initialize the outline generator."""
        self.object_templates = {
            'chair': self._generate_chair_outline,
            'table': self._generate_table_outline,
            'car': self._generate_car_outline,
            'airplane': self._generate_airplane_outline,
            'lamp': self._generate_lamp_outline,
            'sofa': self._generate_sofa_outline,
            'bed': self._generate_bed_outline,
            'desk': self._generate_desk_outline,
            'bookcase': self._generate_bookcase_outline,
            'bicycle': self._generate_bicycle_outline
        }
    
    def generate_outline(self, text: str, num_points: int = 1024) -> List[Tuple[float, float, float]]:
        """
        Generate object outline from text description.
        
        Args:
            text: Text description of the object
            num_points: Number of points to generate
            
        Returns:
            List of (x, y, z) coordinate tuples
        """
        # Extract object type from text
        object_type = self._extract_object_type(text)
        
        # Generate outline based on object type
        if object_type in self.object_templates:
            points = self.object_templates[object_type](text, num_points)
        else:
            # Fallback to generic outline
            points = self._generate_generic_outline(text, num_points)
        
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
            'seat': 'chair',
            'furniture': 'chair',
            'stool': 'chair',
            'bench': 'chair',
            'surface': 'table',
            'desk': 'table',
            'counter': 'table',
            'vehicle': 'car',
            'automobile': 'car',
            'truck': 'car',
            'plane': 'airplane',
            'aircraft': 'airplane',
            'jet': 'airplane',
            'light': 'lamp',
            'lighting': 'lamp',
            'couch': 'sofa',
            'settee': 'sofa',
            'mattress': 'bed',
            'workstation': 'desk',
            'shelf': 'bookcase',
            'bike': 'bicycle',
            'cycle': 'bicycle'
        }
        
        for synonym, obj_type in synonyms.items():
            if synonym in text_lower:
                return obj_type
        
        return 'generic'
    
    def _generate_chair_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate chair outline that actually looks like a chair."""
        points = []
        
        # Seat (thick rectangular surface)
        seat_points = self._generate_rectangle_outline(
            center=(0, 0, 0.1), width=0.5, height=0.5, thickness=0.08, 
            points_per_side=num_points // 6
        )
        points.extend(seat_points)
        
        # Backrest (vertical rectangle with slight angle)
        backrest_points = self._generate_rectangle_outline(
            center=(0, 0.25, 0.5), width=0.5, height=0.6, thickness=0.05,
            points_per_side=num_points // 6
        )
        points.extend(backrest_points)
        
        # Legs (4 vertical lines at corners)
        leg_positions = [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_outline(
                start=(x, y, -0.3), end=(x, y, 0.1), 
                num_points=num_points // 20
            )
            points.extend(leg_points)
        
        # Armrests (if mentioned)
        if 'arm' in text.lower() or 'armrest' in text.lower():
            for side in [-0.25, 0.25]:
                # Armrest support
                support_points = self._generate_line_outline(
                    start=(side, -0.1, 0.1), end=(side, -0.1, 0.3),
                    num_points=num_points // 30
                )
                points.extend(support_points)
                
                # Armrest top
                armrest_points = self._generate_line_outline(
                    start=(side, -0.1, 0.3), end=(side, 0.1, 0.3),
                    num_points=num_points // 30
                )
                points.extend(armrest_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_table_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate table outline."""
        points = []
        
        # Tabletop (thick rectangular surface)
        top_points = self._generate_rectangle_outline(
            center=(0, 0, 0.4), width=0.8, height=0.8, thickness=0.1,
            points_per_side=num_points // 6
        )
        points.extend(top_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)]
        for x, y in leg_positions:
            leg_points = self._generate_line_outline(
                start=(x, y, -0.4), end=(x, y, 0.4),
                num_points=num_points // 16
            )
            points.extend(leg_points)
        
        # Round table variation
        if 'round' in text.lower() or 'circular' in text.lower():
            # Make it more circular
            for i in range(len(points)):
                x, y, z = points[i]
                if z > 0.3:  # Tabletop
                    angle = np.arctan2(y, x)
                    radius = 0.4
                    points[i] = (radius * np.cos(angle), radius * np.sin(angle), z)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_car_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate car outline that actually looks like a car."""
        points = []
        
        # Car body (rectangular with rounded edges)
        body_points = self._generate_rectangle_outline(
            center=(0, 0, 0), width=0.8, height=0.3, thickness=0.2,
            points_per_side=num_points // 8
        )
        points.extend(body_points)
        
        # Hood (front part)
        hood_points = self._generate_rectangle_outline(
            center=(0.3, 0, 0.05), width=0.2, height=0.3, thickness=0.15,
            points_per_side=num_points // 12
        )
        points.extend(hood_points)
        
        # Windshield (angled front)
        windshield_points = self._generate_rectangle_outline(
            center=(0.1, 0, 0.1), width=0.3, height=0.25, thickness=0.02,
            points_per_side=num_points // 15
        )
        points.extend(windshield_points)
        
        # Wheels (4 circles)
        wheel_positions = [(-0.25, -0.15), (0.25, -0.15), (-0.25, 0.15), (0.25, 0.15)]
        for x, y in wheel_positions:
            wheel_points = self._generate_circle_outline(
                center=(x, y, -0.15), radius=0.08, 
                num_points=num_points // 20
            )
            points.extend(wheel_points)
        
        # Doors (side panels)
        door_points = self._generate_rectangle_outline(
            center=(0, 0.15, 0.05), width=0.6, height=0.05, thickness=0.15,
            points_per_side=num_points // 15
        )
        points.extend(door_points)
        
        # Sports car variation
        if 'sports' in text.lower() or 'sleek' in text.lower():
            # Lower profile and longer
            for i in range(len(points)):
                x, y, z = points[i]
                if z > 0:  # Upper body
                    points[i] = (x * 1.2, y, z * 0.6)  # Longer and lower
                else:  # Lower body
                    points[i] = (x * 1.2, y, z)  # Just longer
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_airplane_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate airplane outline that actually looks like an airplane."""
        points = []
        
        # Fuselage (elongated cylinder)
        fuselage_points = self._generate_rectangle_outline(
            center=(0, 0, 0), width=0.8, height=0.1, thickness=0.1,
            points_per_side=num_points // 6
        )
        points.extend(fuselage_points)
        
        # Wings (horizontal rectangles extending outward)
        wing_points = self._generate_rectangle_outline(
            center=(0, 0, 0), width=0.2, height=1.0, thickness=0.03,
            points_per_side=num_points // 8
        )
        points.extend(wing_points)
        
        # Tail (vertical fin)
        tail_points = self._generate_rectangle_outline(
            center=(-0.35, 0, 0.15), width=0.1, height=0.3, thickness=0.05,
            points_per_side=num_points // 10
        )
        points.extend(tail_points)
        
        # Horizontal stabilizers
        stabilizer_points = self._generate_rectangle_outline(
            center=(-0.35, 0, 0.05), width=0.3, height=0.05, thickness=0.02,
            points_per_side=num_points // 15
        )
        points.extend(stabilizer_points)
        
        # Nose (front point)
        nose_points = self._generate_line_outline(
            start=(0.4, 0, 0), end=(0.45, 0, 0),
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
    
    def _generate_lamp_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate lamp outline."""
        points = []
        
        # Base (circular)
        base_points = self._generate_circle_outline(
            center=(0, 0, -0.1), radius=0.2, 
            num_points=num_points // 8
        )
        points.extend(base_points)
        
        # Pole (vertical line)
        pole_points = self._generate_line_outline(
            start=(0, 0, -0.1), end=(0, 0, 0.8),
            num_points=num_points // 4
        )
        points.extend(pole_points)
        
        # Shade (inverted cone)
        shade_points = self._generate_cone_outline(
            center=(0, 0, 0.8), radius=0.3, height=0.2,
            num_points=num_points // 4
        )
        points.extend(shade_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_sofa_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate sofa outline."""
        points = []
        
        # Seat (rectangular surface)
        seat_points = self._generate_rectangle_outline(
            center=(0, 0, 0.1), width=1.2, height=0.6, thickness=0.05,
            points_per_side=num_points // 6
        )
        points.extend(seat_points)
        
        # Backrest (vertical rectangle)
        backrest_points = self._generate_rectangle_outline(
            center=(0, 0.3, 0.4), width=1.2, height=0.6, thickness=0.05,
            points_per_side=num_points // 6
        )
        points.extend(backrest_points)
        
        # Armrests
        for side in [-0.6, 0.6]:
            armrest_points = self._generate_rectangle_outline(
                center=(side, 0, 0.25), width=0.1, height=0.6, thickness=0.05,
                points_per_side=num_points // 12
            )
            points.extend(armrest_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.5, -0.2), (0.5, -0.2), (-0.5, 0.2), (0.5, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_outline(
                start=(x, y, -0.3), end=(x, y, 0.1),
                num_points=num_points // 16
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bed_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bed outline."""
        points = []
        
        # Mattress (rectangular surface)
        mattress_points = self._generate_rectangle_outline(
            center=(0, 0, 0.2), width=1.6, height=0.8, thickness=0.1,
            points_per_side=num_points // 6
        )
        points.extend(mattress_points)
        
        # Headboard (vertical rectangle)
        headboard_points = self._generate_rectangle_outline(
            center=(0, 0.4, 0.6), width=1.6, height=0.8, thickness=0.1,
            points_per_side=num_points // 8
        )
        points.extend(headboard_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3)]
        for x, y in leg_positions:
            leg_points = self._generate_line_outline(
                start=(x, y, -0.2), end=(x, y, 0.2),
                num_points=num_points // 16
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_desk_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate desk outline."""
        points = []
        
        # Desktop (rectangular surface)
        desktop_points = self._generate_rectangle_outline(
            center=(0, 0, 0.3), width=1.0, height=0.6, thickness=0.05,
            points_per_side=num_points // 6
        )
        points.extend(desktop_points)
        
        # Drawers (rectangular box)
        drawer_points = self._generate_rectangle_outline(
            center=(0, -0.3, 0.15), width=0.8, height=0.2, thickness=0.3,
            points_per_side=num_points // 8
        )
        points.extend(drawer_points)
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.4, -0.2), (0.4, -0.2), (-0.4, 0.2), (0.4, 0.2)]
        for x, y in leg_positions:
            leg_points = self._generate_line_outline(
                start=(x, y, -0.2), end=(x, y, 0.3),
                num_points=num_points // 16
            )
            points.extend(leg_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bookcase_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bookcase outline."""
        points = []
        
        # Main structure (vertical rectangle)
        structure_points = self._generate_rectangle_outline(
            center=(0, 0, 0.8), width=0.4, height=1.6, thickness=0.05,
            points_per_side=num_points // 4
        )
        points.extend(structure_points)
        
        # Shelves (horizontal rectangles)
        shelf_heights = [-0.4, 0, 0.4, 0.8, 1.2]
        for height in shelf_heights:
            shelf_points = self._generate_rectangle_outline(
                center=(0, 0, height), width=0.4, height=1.6, thickness=0.02,
                points_per_side=num_points // 20
            )
            points.extend(shelf_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_bicycle_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate bicycle outline."""
        points = []
        
        # Frame (main structure)
        frame_points = self._generate_line_outline(
            start=(-0.3, 0, 0.3), end=(0.3, 0, 0.3),
            num_points=num_points // 8
        )
        points.extend(frame_points)
        
        # Wheels (2 circles)
        wheel_positions = [(-0.3, 0), (0.3, 0)]
        for x, y in wheel_positions:
            wheel_points = self._generate_circle_outline(
                center=(x, y, 0), radius=0.2,
                num_points=num_points // 8
            )
            points.extend(wheel_points)
        
        # Handlebars
        handlebar_points = self._generate_line_outline(
            start=(-0.3, 0, 0.3), end=(-0.3, 0, 0.5),
            num_points=num_points // 16
        )
        points.extend(handlebar_points)
        
        # Seat
        seat_points = self._generate_circle_outline(
            center=(0, 0, 0.3), radius=0.05,
            num_points=num_points // 16
        )
        points.extend(seat_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_generic_outline(self, text: str, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate generic outline for unknown objects."""
        points = []
        
        # Main body (ellipsoid)
        body_points = self._generate_ellipsoid_outline(
            center=(0, 0, 0), radii=(0.3, 0.3, 0.3),
            num_points=num_points // 2
        )
        points.extend(body_points)
        
        # Add some structure
        structure_points = self._generate_rectangle_outline(
            center=(0, 0, 0), width=0.4, height=0.4, thickness=0.4,
            points_per_side=num_points // 4
        )
        points.extend(structure_points)
        
        return points[:num_points] if len(points) > num_points else points
    
    def _generate_rectangle_outline(self, center: Tuple[float, float, float], 
                                  width: float, height: float, thickness: float,
                                  points_per_side: int) -> List[Tuple[float, float, float]]:
        """Generate rectangle outline with wireframe structure."""
        points = []
        cx, cy, cz = center
        
        # Generate points along the edges to create a wireframe
        # Top edge
        for i in range(points_per_side):
            t = i / (points_per_side - 1) if points_per_side > 1 else 0
            x = cx + (t - 0.5) * width
            y = cy + height / 2
            z = cz
            points.append((x, y, z))
        
        # Right edge
        for i in range(points_per_side):
            t = i / (points_per_side - 1) if points_per_side > 1 else 0
            x = cx + width / 2
            y = cy + (t - 0.5) * height
            z = cz
            points.append((x, y, z))
        
        # Bottom edge
        for i in range(points_per_side):
            t = i / (points_per_side - 1) if points_per_side > 1 else 0
            x = cx + (t - 0.5) * width
            y = cy - height / 2
            z = cz
            points.append((x, y, z))
        
        # Left edge
        for i in range(points_per_side):
            t = i / (points_per_side - 1) if points_per_side > 1 else 0
            x = cx - width / 2
            y = cy + (t - 0.5) * height
            z = cz
            points.append((x, y, z))
        
        # Add thickness by creating parallel outlines
        if thickness > 0:
            # Top face
            for i in range(points_per_side // 2):
                t = i / (points_per_side // 2 - 1) if points_per_side // 2 > 1 else 0
                x = cx + (t - 0.5) * width
                y = cy + height / 2
                z = cz + thickness / 2
                points.append((x, y, z))
            
            # Bottom face
            for i in range(points_per_side // 2):
                t = i / (points_per_side // 2 - 1) if points_per_side // 2 > 1 else 0
                x = cx + (t - 0.5) * width
                y = cy - height / 2
                z = cz - thickness / 2
                points.append((x, y, z))
        
        return points
    
    def _generate_line_outline(self, start: Tuple[float, float, float], 
                             end: Tuple[float, float, float], 
                             num_points: int) -> List[Tuple[float, float, float]]:
        """Generate line outline."""
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
    
    def _generate_circle_outline(self, center: Tuple[float, float, float], 
                               radius: float, num_points: int) -> List[Tuple[float, float, float]]:
        """Generate circle outline."""
        points = []
        cx, cy, cz = center
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            z = cz
            points.append((x, y, z))
        
        return points
    
    def _generate_ellipsoid_outline(self, center: Tuple[float, float, float], 
                                  radii: Tuple[float, float, float], 
                                  num_points: int) -> List[Tuple[float, float, float]]:
        """Generate ellipsoid outline."""
        points = []
        cx, cy, cz = center
        rx, ry, rz = radii
        
        # Generate points on the ellipsoid surface
        for i in range(num_points):
            # Use spherical coordinates
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, np.pi)
            
            x = cx + rx * np.sin(v) * np.cos(u)
            y = cy + ry * np.sin(v) * np.sin(u)
            z = cz + rz * np.cos(v)
            
            points.append((x, y, z))
        
        return points
    
    def _generate_cone_outline(self, center: Tuple[float, float, float], 
                             radius: float, height: float, 
                             num_points: int) -> List[Tuple[float, float, float]]:
        """Generate cone outline."""
        points = []
        cx, cy, cz = center
        
        # Base circle
        base_points = self._generate_circle_outline(
            center=(cx, cy, cz), radius=radius, num_points=num_points // 2
        )
        points.extend(base_points)
        
        # Cone sides
        for i in range(num_points // 2):
            angle = 2 * np.pi * i / (num_points // 2)
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            z = cz
            points.append((x, y, z))
            
            # Top point
            points.append((cx, cy, cz + height))

# Test function
def test_outline_generator():
    """Test the outline generator."""
    generator = ObjectOutlineGenerator()
    
    test_texts = [
        "a wooden chair with four legs",
        "a red sports car with two doors",
        "a white commercial airplane",
        "a modern glass table",
        "a comfortable leather sofa"
    ]
    
    for text in test_texts:
        print(f"\nGenerating outline for: '{text}'")
        points = generator.generate_outline(text, 200)
        print(f"Generated {len(points)} points")
        
        # Show first few points
        print(f"First 5 points: {points[:5]}")

if __name__ == "__main__":
    test_outline_generator()
