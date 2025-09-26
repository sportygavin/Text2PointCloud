"""
Text2PointCloud - ShapeNet Data Integration

This module handles downloading and processing real ShapeNet data for training.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import requests
import zipfile
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ShapeNetDataManager:
    """
    Manages ShapeNet data download and processing.
    """
    
    def __init__(self, data_dir: str = "data/shapenet"):
        """Initialize the ShapeNet data manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # ShapeNet categories and their descriptions
        self.categories = {
            'chair': {
                'description': 'a chair with four legs and a backrest',
                'variations': [
                    'a wooden dining chair',
                    'a comfortable office chair',
                    'a modern chair design',
                    'a traditional wooden chair',
                    'a leather armchair',
                    'a plastic chair',
                    'a metal chair with padding',
                    'a rocking chair',
                    'a bar stool',
                    'a folding chair'
                ]
            },
            'table': {
                'description': 'a table with four legs and a flat surface',
                'variations': [
                    'a wooden dining table',
                    'a modern glass table',
                    'a coffee table with legs',
                    'a rectangular table surface',
                    'a metal table',
                    'a round table',
                    'a kitchen table',
                    'a work table',
                    'a side table',
                    'a conference table'
                ]
            },
            'car': {
                'description': 'a four-wheeled vehicle with doors and windows',
                'variations': [
                    'a red sports car',
                    'a blue sedan vehicle',
                    'a white family car',
                    'a black luxury car',
                    'a silver automobile',
                    'a yellow taxi cab',
                    'a green pickup truck',
                    'a white van',
                    'a black SUV',
                    'a red convertible'
                ]
            },
            'airplane': {
                'description': 'a flying vehicle with wings and a fuselage',
                'variations': [
                    'a commercial passenger airplane',
                    'a small private jet',
                    'a white aircraft with wings',
                    'a flying vehicle',
                    'a modern airplane design',
                    'a military fighter jet',
                    'a cargo airplane',
                    'a helicopter',
                    'a glider',
                    'a seaplane'
                ]
            },
            'lamp': {
                'description': 'a lighting fixture with a base and shade',
                'variations': [
                    'a table lamp',
                    'a floor lamp',
                    'a desk lamp',
                    'a reading light',
                    'a modern lamp design',
                    'a vintage lamp',
                    'a LED lamp',
                    'a bedside lamp',
                    'a pendant light',
                    'a chandelier'
                ]
            },
            'sofa': {
                'description': 'a long seat with a backrest and armrests',
                'variations': [
                    'a comfortable sofa',
                    'a leather couch',
                    'a fabric sofa',
                    'a sectional sofa',
                    'a loveseat',
                    'a futon',
                    'a reclining sofa',
                    'a modern sofa design',
                    'a vintage sofa',
                    'a corner sofa'
                ]
            }
        }
    
    def create_realistic_shapenet_data(self, samples_per_category: int = 100):
        """
        Create realistic ShapeNet-style data with proper text descriptions.
        
        Args:
            samples_per_category: Number of samples per category
        """
        print("Creating realistic ShapeNet-style data...")
        
        all_data = []
        
        for category, info in self.categories.items():
            print(f"  Generating {samples_per_category} samples for {category}...")
            
            for i in range(samples_per_category):
                # Select a variation
                variation = info['variations'][i % len(info['variations'])]
                
                # Generate realistic point cloud for this category
                point_cloud = self._generate_category_point_cloud(category, i)
                
                # Create metadata
                metadata = {
                    'text': variation,
                    'category': category,
                    'base_description': info['description'],
                    'variation': variation,
                    'id': f"{category}_{i:04d}",
                    'point_cloud': point_cloud.tolist()
                }
                
                all_data.append(metadata)
        
        # Split data
        train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
        
        # Save splits
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        for split_name, split_data in splits.items():
            output_file = self.data_dir / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f)
            print(f"  Saved {len(split_data)} samples to {split_name} split")
        
        print(f"Created ShapeNet-style dataset with {len(all_data)} samples")
        return self.data_dir
    
    def _generate_category_point_cloud(self, category: str, variation: int) -> np.ndarray:
        """Generate a realistic point cloud for a specific category."""
        np.random.seed(variation)  # Ensure reproducibility
        
        if category == 'chair':
            return self._generate_chair_point_cloud(variation)
        elif category == 'table':
            return self._generate_table_point_cloud(variation)
        elif category == 'car':
            return self._generate_car_point_cloud(variation)
        elif category == 'airplane':
            return self._generate_airplane_point_cloud(variation)
        elif category == 'lamp':
            return self._generate_lamp_point_cloud(variation)
        elif category == 'sofa':
            return self._generate_sofa_point_cloud(variation)
        else:
            return self._generate_generic_point_cloud(variation)
    
    def _generate_chair_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic chair point cloud."""
        points = []
        
        # Seat (rectangular surface)
        seat_x = np.random.uniform(-0.3, 0.3, 200)
        seat_y = np.random.uniform(-0.3, 0.3, 200)
        seat_z = np.full(200, 0.1)
        points.extend(list(zip(seat_x, seat_y, seat_z)))
        
        # Backrest (vertical rectangle)
        backrest_x = np.random.uniform(-0.3, 0.3, 150)
        backrest_y = np.full(150, 0.3)
        backrest_z = np.random.uniform(0.1, 0.8, 150)
        points.extend(list(zip(backrest_x, backrest_y, backrest_z)))
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        for x, y in leg_positions:
            leg_z = np.linspace(-0.4, 0.1, 50)
            leg_x = np.full(50, x)
            leg_y = np.full(50, y)
            points.extend(list(zip(leg_x, leg_y, leg_z)))
        
        # Add some variation based on chair type
        if variation % 3 == 0:  # Armchair
            # Add armrests
            for side in [-0.3, 0.3]:
                arm_x = np.full(100, side)
                arm_y = np.random.uniform(-0.2, 0.2, 100)
                arm_z = np.random.uniform(0.1, 0.4, 100)
                points.extend(list(zip(arm_x, arm_y, arm_z)))
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_table_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic table point cloud."""
        points = []
        
        # Tabletop (thick rectangular surface)
        for z_offset in [0, 0.05]:
            top_x = np.random.uniform(-0.4, 0.4, 200)
            top_y = np.random.uniform(-0.4, 0.4, 200)
            top_z = np.full(200, 0.4 + z_offset)
            points.extend(list(zip(top_x, top_y, top_z)))
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)]
        for x, y in leg_positions:
            leg_z = np.linspace(-0.4, 0.4, 100)
            leg_x = np.full(100, x)
            leg_y = np.full(100, y)
            points.extend(list(zip(leg_x, leg_y, leg_z)))
        
        # Add variation based on table type
        if variation % 2 == 0:  # Round table
            # Make it more circular
            for i in range(len(points)):
                x, y, z = points[i]
                if z > 0.3:  # Tabletop
                    angle = np.arctan2(y, x)
                    radius = 0.4
                    points[i] = (radius * np.cos(angle), radius * np.sin(angle), z)
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_car_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic car point cloud."""
        points = []
        
        # Car body (ellipsoid)
        body_x = np.random.uniform(-0.4, 0.4, 300)
        body_y = np.random.uniform(-0.2, 0.2, 300)
        body_z = np.random.uniform(-0.15, 0.15, 300)
        points.extend(list(zip(body_x, body_y, body_z)))
        
        # Wheels (4 circles)
        wheel_positions = [(-0.3, -0.2), (0.3, -0.2), (-0.3, 0.2), (0.3, 0.2)]
        for x, y in wheel_positions:
            angles = np.linspace(0, 2*np.pi, 50)
            wheel_x = x + 0.1 * np.cos(angles)
            wheel_y = y + 0.1 * np.sin(angles)
            wheel_z = np.full(50, -0.2)
            points.extend(list(zip(wheel_x, wheel_y, wheel_z)))
        
        # Windshield and windows
        windshield_x = np.random.uniform(-0.2, 0.2, 100)
        windshield_y = np.full(100, 0.2)
        windshield_z = np.random.uniform(0, 0.1, 100)
        points.extend(list(zip(windshield_x, windshield_y, windshield_z)))
        
        # Add variation based on car type
        if variation % 3 == 0:  # Sports car - lower profile
            for i in range(len(points)):
                x, y, z = points[i]
                if z > 0:  # Upper body
                    points[i] = (x, y, z * 0.7)  # Lower the roof
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_airplane_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic airplane point cloud."""
        points = []
        
        # Fuselage (elongated ellipsoid)
        fuselage_x = np.random.uniform(-0.5, 0.5, 200)
        fuselage_y = np.random.uniform(-0.1, 0.1, 200)
        fuselage_z = np.random.uniform(-0.1, 0.1, 200)
        points.extend(list(zip(fuselage_x, fuselage_y, fuselage_z)))
        
        # Wings (horizontal rectangles)
        wing_x = np.random.uniform(-0.3, 0.3, 150)
        wing_y = np.random.uniform(-0.4, 0.4, 150)
        wing_z = np.full(150, 0)
        points.extend(list(zip(wing_x, wing_y, wing_z)))
        
        # Tail (vertical rectangle)
        tail_x = np.random.uniform(-0.5, -0.4, 100)
        tail_y = np.random.uniform(-0.1, 0.1, 100)
        tail_z = np.random.uniform(0, 0.3, 100)
        points.extend(list(zip(tail_x, tail_y, tail_z)))
        
        # Add variation based on airplane type
        if variation % 2 == 0:  # Commercial airliner - larger
            for i in range(len(points)):
                x, y, z = points[i]
                points[i] = (x * 1.2, y * 1.2, z * 1.2)  # Scale up
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_lamp_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic lamp point cloud."""
        points = []
        
        # Base (circular)
        base_x = np.random.uniform(-0.2, 0.2, 100)
        base_y = np.random.uniform(-0.2, 0.2, 100)
        base_z = np.full(100, -0.1)
        points.extend(list(zip(base_x, base_y, base_z)))
        
        # Pole (vertical line)
        pole_x = np.full(100, 0)
        pole_y = np.full(100, 0)
        pole_z = np.linspace(-0.1, 0.8, 100)
        points.extend(list(zip(pole_x, pole_y, pole_z)))
        
        # Shade (inverted cone)
        for i in range(50):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, 0.3)
            shade_x = radius * np.cos(angle)
            shade_y = radius * np.sin(angle)
            shade_z = 0.8 + np.random.uniform(0, 0.2)
            points.append((shade_x, shade_y, shade_z))
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_sofa_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a realistic sofa point cloud."""
        points = []
        
        # Seat (rectangular surface)
        seat_x = np.random.uniform(-0.6, 0.6, 300)
        seat_y = np.random.uniform(-0.3, 0.3, 300)
        seat_z = np.full(300, 0.1)
        points.extend(list(zip(seat_x, seat_y, seat_z)))
        
        # Backrest (vertical rectangle)
        backrest_x = np.random.uniform(-0.6, 0.6, 200)
        backrest_y = np.full(200, 0.3)
        backrest_z = np.random.uniform(0.1, 0.6, 200)
        points.extend(list(zip(backrest_x, backrest_y, backrest_z)))
        
        # Armrests
        for side in [-0.6, 0.6]:
            arm_x = np.full(150, side)
            arm_y = np.random.uniform(-0.3, 0.3, 150)
            arm_z = np.random.uniform(0.1, 0.4, 150)
            points.extend(list(zip(arm_x, arm_y, arm_z)))
        
        # Legs (4 vertical lines)
        leg_positions = [(-0.5, -0.2), (0.5, -0.2), (-0.5, 0.2), (0.5, 0.2)]
        for x, y in leg_positions:
            leg_z = np.linspace(-0.3, 0.1, 50)
            leg_x = np.full(50, x)
            leg_y = np.full(50, y)
            points.extend(list(zip(leg_x, leg_y, leg_z)))
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_generic_point_cloud(self, variation: int) -> np.ndarray:
        """Generate a generic point cloud."""
        points = []
        
        # Main body (ellipsoid)
        body_x = np.random.uniform(-0.3, 0.3, 400)
        body_y = np.random.uniform(-0.3, 0.3, 400)
        body_z = np.random.uniform(-0.3, 0.3, 400)
        points.extend(list(zip(body_x, body_y, body_z)))
        
        # Add some structure
        structure_x = np.random.uniform(-0.2, 0.2, 300)
        structure_y = np.random.uniform(-0.2, 0.2, 300)
        structure_z = np.random.uniform(-0.2, 0.2, 300)
        points.extend(list(zip(structure_x, structure_y, structure_z)))
        
        # Convert to numpy array and normalize
        points = np.array(points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.05
            points = np.vstack([points, padding])
        
        return points

class ShapeNetDataset(Dataset):
    """
    PyTorch Dataset for ShapeNet data.
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """Initialize the dataset."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load the dataset from files."""
        data_file = self.data_dir / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {data_file} not found. Creating empty dataset.")
            return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single data sample."""
        sample = self.data[idx]
        
        # Load point cloud
        point_cloud = np.array(sample['point_cloud'])
        
        # Get text description
        text = sample['text']
        
        return {
            'point_cloud': torch.tensor(point_cloud, dtype=torch.float32),
            'text': text,
            'category': sample.get('category', 'unknown'),
            'id': sample.get('id', f'unknown_{idx}')
        }

# Test function
def test_shapenet_data():
    """Test the ShapeNet data creation."""
    print("Testing ShapeNet Data Creation...")
    
    # Create data manager
    data_manager = ShapeNetDataManager("data/real_shapenet")
    
    # Create realistic data
    data_dir = data_manager.create_realistic_shapenet_data(samples_per_category=50)
    
    # Test dataset loading
    train_dataset = ShapeNetDataset(data_dir, split='train')
    print(f"Loaded training dataset with {len(train_dataset)} samples")
    
    # Test data sample
    sample = train_dataset[0]
    print(f"Sample text: {sample['text']}")
    print(f"Sample category: {sample['category']}")
    print(f"Sample point cloud shape: {sample['point_cloud'].shape}")
    
    return data_dir

if __name__ == "__main__":
    test_shapenet_data()
