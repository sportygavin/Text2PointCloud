"""
Text2PointCloud - Model Training System

This module handles training the ML model on text-point cloud pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from src.training.data_manager import PointCloudDataset
from src.shape_generator.ml_generator import PointCloudGenerator

class TextEncoder(nn.Module):
    """
    Text encoder using a simple neural network.
    In practice, you'd use CLIP or similar pre-trained encoder.
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 512)  # Output dimension
        )
    
    def forward(self, text_tokens):
        embedded = self.embedding(text_tokens)
        # Simple pooling (in practice, use more sophisticated methods)
        pooled = torch.mean(embedded, dim=1)
        return self.encoder(pooled)

class TextToPointCloudTrainer:
    """
    Trainer for the text-to-point-cloud model.
    """
    
    def __init__(self, model: PointCloudGenerator, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            point_clouds = batch['point_cloud'].to(self.device)
            texts = batch['text']
            
            # Convert text to simple embeddings (mock)
            text_embeddings = self._text_to_embedding(texts)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_points = self.model(text_embeddings)
            
            # Calculate loss
            loss = criterion(predicted_points, point_clouds)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                point_clouds = batch['point_cloud'].to(self.device)
                texts = batch['text']
                
                text_embeddings = self._text_to_embedding(texts)
                predicted_points = self.model(text_embeddings)
                
                loss = criterion(predicted_points, point_clouds)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _text_to_embedding(self, texts: List[str]) -> torch.Tensor:
        """Convert text to embeddings (mock implementation)."""
        # Simple hash-based embedding for demonstration
        embeddings = []
        for text in texts:
            text_hash = abs(hash(text.lower())) % (2**32 - 1)
            embedding = np.random.RandomState(text_hash).normal(0, 1, 512)
            embeddings.append(embedding)
        
        return torch.tensor(np.array(embeddings), dtype=torch.float32).to(self.device)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 10, learning_rate: float = 0.001):
        """Train the model."""
        # Setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print(f"Starting training for {epochs} epochs...")
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
            print('-' * 50)
    
    def plot_training_history(self):
        """Plot training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
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

def test_training():
    """Test the training system."""
    print("Testing Model Training System...")
    
    # Create data manager
    from src.training.data_manager import TrainingDataManager
    data_manager = TrainingDataManager("test_data")
    
    # Create synthetic dataset
    categories = ['chair', 'table', 'car']
    data_manager.create_synthetic_dataset(categories, samples_per_category=50)
    
    # Load datasets
    train_dataset = PointCloudDataset("test_data/splits", split='train')
    val_dataset = PointCloudDataset("test_data/splits", split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    model = PointCloudGenerator(text_dim=512, point_dim=3, num_points=1024)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TextToPointCloudTrainer(model, device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=5, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model("test_data/trained_model.pth")

if __name__ == "__main__":
    test_training()
