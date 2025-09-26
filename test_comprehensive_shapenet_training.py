"""
Text2PointCloud - Comprehensive Training with ShapeNet Data

This module provides a complete training pipeline using ShapeNet data
and improved text encoding (without CLIP dependency issues).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from src.training.shapenet_data import ShapeNetDataManager, ShapeNetDataset
from src.text_processing.improved_encoder import ImprovedTextEncoder
from src.validation.validator import PointCloudValidator

class TextToPointCloudModel(nn.Module):
    """
    Complete text-to-point-cloud model using improved text encoding.
    """
    
    def __init__(self, text_encoder: ImprovedTextEncoder, point_dim: int = 3, num_points: int = 1024):
        super().__init__()
        self.text_encoder = text_encoder
        self.point_dim = point_dim
        self.num_points = num_points
        self.text_dim = 512  # Improved encoder output dimension
        
        # Point cloud generation network
        self.point_decoder = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_points * point_dim)
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
            Point cloud tensor of shape (batch_size, num_points, point_dim)
        """
        # Encode text
        text_embeddings = []
        for text in text_inputs:
            embedding = self.text_encoder.encode_text(text)
            text_embeddings.append(embedding.squeeze(0))
        
        text_embeddings = torch.stack(text_embeddings)
        
        # Generate point clouds
        point_clouds = self.point_decoder(text_embeddings)
        
        # Reshape to point cloud format
        point_clouds = point_clouds.view(-1, self.num_points, self.point_dim)
        
        return point_clouds

class ComprehensiveTrainer:
    """
    Comprehensive trainer for the text-to-point-cloud model.
    """
    
    def __init__(self, model: TextToPointCloudModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Validator
        self.validator = PointCloudValidator()
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            point_clouds = batch['point_cloud'].to(self.device)
            texts = batch['text']
            
            # Forward pass
            optimizer.zero_grad()
            predicted_point_clouds = self.model(texts)
            
            # Calculate loss
            loss = criterion(predicted_point_clouds, point_clouds)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                point_clouds = batch['point_cloud'].to(self.device)
                texts = batch['text']
                
                predicted_point_clouds = self.model(texts)
                loss = criterion(predicted_point_clouds, point_clouds)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, learning_rate: float = 0.001):
        """Train the model."""
        # Setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print(f"Starting comprehensive training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.epochs.append(epoch)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Test generation every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._test_generation(epoch + 1)
            
            print('-' * 50)
    
    def _test_generation(self, epoch: int):
        """Test point cloud generation during training."""
        print(f"  Testing generation at epoch {epoch}:")
        
        test_texts = [
            "a wooden chair with four legs",
            "a red sports car with two doors",
            "a white commercial airplane",
            "a modern glass table"
        ]
        
        self.model.eval()
        with torch.no_grad():
            generated_point_clouds = self.model(test_texts)
            
            for i, text in enumerate(test_texts):
                points = generated_point_clouds[i].cpu().numpy()
                
                # Validate
                validation_results = self.validator.validate_generation(points, text)
                
                predicted_category = validation_results.get('category_accuracy', {}).get('predicted_category', 'unknown')
                quality = validation_results.get('quality_metrics', {}).get('compactness', 0.0)
                
                print(f"    '{text}': Category={predicted_category}, Quality={quality:.3f}")
    
    def plot_training_history(self):
        """Plot training history."""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss ratio plot
        plt.subplot(1, 3, 2)
        if len(self.val_losses) > 0:
            loss_ratios = [t/v if v > 0 else 0 for t, v in zip(self.train_losses, self.val_losses)]
            plt.plot(self.epochs, loss_ratios, label='Train/Val Loss Ratio', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.title('Overfitting Check')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss improvement plot
        plt.subplot(1, 3, 3)
        if len(self.train_losses) > 1:
            train_improvement = [self.train_losses[0] - loss for loss in self.train_losses]
            val_improvement = [self.val_losses[0] - loss for loss in self.val_losses]
            plt.plot(self.epochs, train_improvement, label='Train Improvement', marker='o')
            plt.plot(self.epochs, val_improvement, label='Val Improvement', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Improvement')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'text_encoder_vocab': self.model.text_encoder.vocab,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.epochs = checkpoint.get('epochs', [])
        print(f"Model loaded from {path}")

def create_training_data():
    """Create comprehensive training data."""
    print("Creating comprehensive ShapeNet-style training data...")
    
    # Create data manager
    data_manager = ShapeNetDataManager("data/comprehensive_shapenet")
    
    # Create realistic data
    data_dir = data_manager.create_realistic_shapenet_data(samples_per_category=200)
    
    return data_dir

def test_comprehensive_training():
    """Test the comprehensive training system."""
    print("Testing Comprehensive Training with ShapeNet Data...")
    
    # Create training data
    data_dir = create_training_data()
    
    # Load datasets
    train_dataset = ShapeNetDataset(data_dir, split='train')
    val_dataset = ShapeNetDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create model
    text_encoder = ImprovedTextEncoder()
    model = TextToPointCloudModel(text_encoder)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ComprehensiveTrainer(model, device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=30, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model("data/comprehensive_trained_model.pth")
    
    # Test final generation
    print("\nFinal generation test:")
    test_texts = [
        "a comfortable wooden dining chair",
        "a sleek red sports car",
        "a large white commercial airplane",
        "a modern glass coffee table"
    ]
    
    model.eval()
    with torch.no_grad():
        generated_point_clouds = model(test_texts)
        
        for i, text in enumerate(test_texts):
            points = generated_point_clouds[i].cpu().numpy()
            validation_results = trainer.validator.validate_generation(points, text)
            
            print(f"  '{text}':")
            print(f"    Category: {validation_results['category_accuracy']['predicted_category']}")
            print(f"    Quality: {validation_results['quality_metrics']['compactness']:.3f}")
            print(f"    Dimensions: {[f'{d:.2f}' for d in validation_results['shape_metrics']['dimensions']]}")

if __name__ == "__main__":
    test_comprehensive_training()
