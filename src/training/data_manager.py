"""
Text2PointCloud - Training Data Management

This module handles training data collection and management for ML models.
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

class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for text-point cloud pairs.
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: 'train', 'val', or 'test'
        """
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
            'category': sample.get('category', 'unknown')
        }

class TrainingDataManager:
    """
    Manages training data collection and preparation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "splits").mkdir(exist_ok=True)
    
    def download_shapenet(self, category: str = "chair"):
        """
        Download ShapeNet data for a specific category.
        
        Args:
            category: ShapeNet category (e.g., 'chair', 'table', 'car')
        """
        print(f"Downloading ShapeNet {category} data...")
        
        # ShapeNet download URLs (these would need to be updated with actual URLs)
        # For now, we'll create a mock download function
        self._mock_download_shapenet(category)
    
    def _mock_download_shapenet(self, category: str):
        """Mock function to simulate ShapeNet download."""
        print(f"Mock: Downloading {category} data from ShapeNet...")
        
        # Create mock data
        mock_data = []
        for i in range(100):  # Mock 100 samples
            # Generate random point cloud
            points = np.random.randn(1024, 3) * 0.5
            
            # Create text description
            descriptions = {
                'chair': f"a {np.random.choice(['wooden', 'metal', 'plastic', 'leather'])} chair",
                'table': f"a {np.random.choice(['wooden', 'glass', 'metal'])} table",
                'car': f"a {np.random.choice(['red', 'blue', 'black', 'white'])} car",
                'airplane': f"a {np.random.choice(['passenger', 'cargo', 'fighter'])} airplane"
            }
            
            text = descriptions.get(category, f"a {category}")
            
            mock_data.append({
                'point_cloud': points.tolist(),
                'text': text,
                'category': category,
                'id': f"{category}_{i:03d}"
            })
        
        # Save mock data
        output_file = self.data_dir / "raw" / f"{category}_mock.json"
        with open(output_file, 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        print(f"Mock data saved to {output_file}")
    
    def create_synthetic_dataset(self, categories: List[str], samples_per_category: int = 1000):
        """
        Create a synthetic dataset with text-point cloud pairs.
        
        Args:
            categories: List of object categories
            samples_per_category: Number of samples per category
        """
        print("Creating synthetic dataset...")
        
        all_data = []
        
        for category in categories:
            print(f"Generating {samples_per_category} samples for {category}...")
            
            for i in range(samples_per_category):
                # Generate point cloud based on category
                point_cloud = self._generate_category_point_cloud(category)
                
                # Generate text description
                text = self._generate_text_description(category)
                
                all_data.append({
                    'point_cloud': point_cloud.tolist(),
                    'text': text,
                    'category': category,
                    'id': f"{category}_{i:04d}"
                })
        
        # Split data
        self._split_and_save_data(all_data)
        print(f"Created dataset with {len(all_data)} samples")
    
    def _generate_category_point_cloud(self, category: str) -> np.ndarray:
        """Generate a point cloud for a specific category."""
        if category == 'chair':
            return self._generate_chair_point_cloud()
        elif category == 'table':
            return self._generate_table_point_cloud()
        elif category == 'car':
            return self._generate_car_point_cloud()
        else:
            # Default: random point cloud
            return np.random.randn(1024, 3) * 0.5
    
    def _generate_chair_point_cloud(self) -> np.ndarray:
        """Generate a chair-like point cloud."""
        points = []
        
        # Seat (rectangle)
        seat_points = np.random.uniform(-0.3, 0.3, (200, 2))
        seat_points = np.column_stack([seat_points, np.zeros(200)])
        points.extend(seat_points)
        
        # Backrest (rectangle)
        backrest_points = np.random.uniform(-0.3, 0.3, (200, 2))
        backrest_points = np.column_stack([backrest_points, np.random.uniform(0.2, 0.8, 200)])
        points.extend(backrest_points)
        
        # Legs (4 vertical lines)
        for x in [-0.25, 0.25]:
            for y in [-0.25, 0.25]:
                leg_points = np.column_stack([
                    np.full(50, x),
                    np.full(50, y),
                    np.linspace(-0.4, 0, 50)
                ])
                points.extend(leg_points)
        
        # Convert to numpy array and pad/truncate to 1024 points
        points = np.array(points)
        if len(points) > 1024:
            points = points[:1024]
        else:
            # Pad with random points
            padding = np.random.randn(1024 - len(points), 3) * 0.1
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_table_point_cloud(self) -> np.ndarray:
        """Generate a table-like point cloud."""
        points = []
        
        # Tabletop (rectangle)
        top_points = np.random.uniform(-0.4, 0.4, (300, 2))
        top_points = np.column_stack([top_points, np.full(300, 0.4)])
        points.extend(top_points)
        
        # Legs (4 vertical lines)
        for x in [-0.3, 0.3]:
            for y in [-0.3, 0.3]:
                leg_points = np.column_stack([
                    np.full(100, x),
                    np.full(100, y),
                    np.linspace(-0.4, 0.4, 100)
                ])
                points.extend(leg_points)
        
        # Convert to numpy array and pad/truncate to 1024 points
        points = np.array(points)
        if len(points) > 1024:
            points = points[:1024]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.1
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_car_point_cloud(self) -> np.ndarray:
        """Generate a car-like point cloud."""
        points = []
        
        # Car body (ellipsoid)
        body_points = np.random.uniform(-0.4, 0.4, (400, 3))
        body_points[:, 2] *= 0.3  # Flatten in Z
        points.extend(body_points)
        
        # Wheels (4 circles)
        for x in [-0.3, 0.3]:
            for y in [-0.2, 0.2]:
                wheel_points = np.random.uniform(-0.1, 0.1, (100, 2))
                wheel_points = np.column_stack([
                    wheel_points[:, 0] + x,
                    wheel_points[:, 1] + y,
                    np.full(100, -0.3)
                ])
                points.extend(wheel_points)
        
        # Convert to numpy array and pad/truncate to 1024 points
        points = np.array(points)
        if len(points) > 1024:
            points = points[:1024]
        else:
            padding = np.random.randn(1024 - len(points), 3) * 0.1
            points = np.vstack([points, padding])
        
        return points
    
    def _generate_text_description(self, category: str) -> str:
        """Generate a text description for a category."""
        descriptions = {
            'chair': f"a {np.random.choice(['wooden', 'metal', 'plastic', 'leather'])} chair",
            'table': f"a {np.random.choice(['wooden', 'glass', 'metal'])} table",
            'car': f"a {np.random.choice(['red', 'blue', 'black', 'white'])} car",
            'airplane': f"a {np.random.choice(['passenger', 'cargo', 'fighter'])} airplane"
        }
        return descriptions.get(category, f"a {category}")
    
    def _split_and_save_data(self, data: List[Dict]):
        """Split data into train/val/test and save."""
        # Shuffle data
        np.random.shuffle(data)
        
        # Split: 70% train, 15% val, 15% test
        n = len(data)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        splits = {
            'train': data[:train_end],
            'val': data[train_end:val_end],
            'test': data[val_end:]
        }
        
        # Save splits
        for split_name, split_data in splits.items():
            output_file = self.data_dir / "splits" / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {len(split_data)} samples to {split_name} split")

# Example usage
def test_training_data():
    """Test the training data management system."""
    print("Testing Training Data Management...")
    
    # Create data manager
    data_manager = TrainingDataManager("test_data")
    
    # Create synthetic dataset
    categories = ['chair', 'table', 'car']
    data_manager.create_synthetic_dataset(categories, samples_per_category=100)
    
    # Test dataset loading
    dataset = PointCloudDataset("test_data/splits", split='train')
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Test data loading
    sample = dataset[0]
    print(f"Sample point cloud shape: {sample['point_cloud'].shape}")
    print(f"Sample text: {sample['text']}")
    print(f"Sample category: {sample['category']}")

if __name__ == "__main__":
    test_training_data()
