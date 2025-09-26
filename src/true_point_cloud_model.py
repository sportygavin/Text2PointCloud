"""
Text2PointCloud - True Point Cloud Model

This model generates actual point cloud outlines using the true point cloud generator.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.improved_encoder import ImprovedTextEncoder
from true_point_cloud_generator import TruePointCloudGenerator

class TruePointCloudModel(nn.Module):
    """
    Model that generates actual point cloud outlines using the true point cloud generator.
    """
    
    def __init__(self, text_encoder: ImprovedTextEncoder, num_points: int = 1024):
        super().__init__()
        self.text_encoder = text_encoder
        self.num_points = num_points
        self.point_cloud_generator = TruePointCloudGenerator()
        
        # Text embedding dimension
        self.text_dim = 512
        
        # Point cloud refinement network
        self.refinement_net = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # x, y, z adjustments
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_inputs: List[str]):
        """
        Generate point clouds from text inputs.
        
        Args:
            text_inputs: List of text descriptions
            
        Returns:
            Point cloud tensor of shape (batch_size, num_points, 3)
        """
        batch_size = len(text_inputs)
        point_clouds = []
        
        for text in text_inputs:
            # Generate base point cloud using true generator
            base_points = self.point_cloud_generator.generate_point_cloud(text, self.num_points)
            
            # Ensure we have exactly num_points
            if len(base_points) < self.num_points:
                # Pad with random points
                while len(base_points) < self.num_points:
                    base_points.append((0, 0, 0))
            elif len(base_points) > self.num_points:
                # Truncate to num_points
                base_points = base_points[:self.num_points]
            
            # Convert to tensor
            base_tensor = torch.tensor(base_points, dtype=torch.float32)
            
            # Encode text
            text_embedding = self.text_encoder.encode_text(text)
            
            # Generate refinement adjustments
            adjustments = self.refinement_net(text_embedding.squeeze(0))
            
            # Apply adjustments to base points
            refined_points = base_tensor + adjustments.unsqueeze(0).expand(self.num_points, -1)
            
            point_clouds.append(refined_points)
        
        return torch.stack(point_clouds)
    
    def generate_point_cloud(self, text: str) -> List[Tuple[float, float, float]]:
        """
        Generate point cloud from text description.
        
        Args:
            text: Text description
            
        Returns:
            List of (x, y, z) coordinate tuples
        """
        self.eval()
        with torch.no_grad():
            point_cloud = self.forward([text])
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
            
            # Convert to list of tuples
            points = [(float(x), float(y), float(z)) for x, y, z in point_cloud]
            
            return points

def test_true_point_cloud_model():
    """Test the true point cloud model."""
    print("Testing True Point Cloud Model...")
    
    # Create model
    text_encoder = ImprovedTextEncoder()
    model = TruePointCloudModel(text_encoder)
    
    # Test texts
    test_texts = [
        "a wooden chair with four legs",
        "a red sports car with two doors",
        "a white commercial airplane",
        "a modern glass table"
    ]
    
    for text in test_texts:
        print(f"\nGenerating: '{text}'")
        points = model.generate_point_cloud(text)
        print(f"Generated {len(points)} points")
        print(f"First 3 points: {points[:3]}")
        
        # Check if points look like an outline
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")

if __name__ == "__main__":
    test_true_point_cloud_model()
