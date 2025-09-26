"""
Text2PointCloud - Comprehensive Training System

This module provides a complete training pipeline that will actually teach
the model to generate different point clouds for different text inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from src.text_processing.improved_encoder import ImprovedTextEncoder
from src.validation.validator import PointCloudValidator

class TextToPointCloudModel(nn.Module):
    """
    Complete text-to-point-cloud model with proper architecture.
    """
    
    def __init__(self, text_encoder: ImprovedTextEncoder, point_dim: int = 3, num_points: int = 1024):
        super().__init__()
        self.text_encoder = text_encoder
        self.point_dim = point_dim
        self.num_points = num_points
        
        # Text processing
        self.text_dim = 512
        
        # Point cloud generation network
        self.point_decoder = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_points * point_dim)
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

class TrainingDataset(Dataset):
    """
    Dataset for training the text-to-point-cloud model.
    """
    
    def __init__(self, data: List[Dict]):
        """Initialize the dataset."""
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single data sample."""
        sample = self.data[idx]
        
        return {
            'text': sample['text'],
            'point_cloud': torch.tensor(sample['point_cloud'], dtype=torch.float32),
            'category': sample['category']
        }

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
            texts = batch['text']
            target_point_clouds = batch['point_cloud'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_point_clouds = self.model(texts)
            
            # Calculate loss
            loss = criterion(predicted_point_clouds, target_point_clouds)
            
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
                texts = batch['text']
                target_point_clouds = batch['point_cloud'].to(self.device)
                
                predicted_point_clouds = self.model(texts)
                loss = criterion(predicted_point_clouds, target_point_clouds)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 20, learning_rate: float = 0.001):
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
            
            # Test generation every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._test_generation(epoch + 1)
            
            print('-' * 50)
    
    def _test_generation(self, epoch: int):
        """Test point cloud generation during training."""
        print(f"  Testing generation at epoch {epoch}:")
        
        test_texts = [
            "a wooden chair",
            "a red car", 
            "a white airplane",
            "a glass table"
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
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss ratio plot
        plt.subplot(1, 2, 2)
        if len(self.val_losses) > 0:
            loss_ratios = [t/v if v > 0 else 0 for t, v in zip(self.train_losses, self.val_losses)]
            plt.plot(self.epochs, loss_ratios, label='Train/Val Loss Ratio', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.title('Overfitting Check')
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
    print("Creating comprehensive training data...")
    
    # Create realistic training data
    categories = {
        'chair': [
            'a wooden chair with four legs',
            'a comfortable office chair',
            'a dining chair with backrest',
            'a modern chair design',
            'a traditional wooden chair'
        ],
        'table': [
            'a wooden dining table',
            'a modern glass table',
            'a coffee table with legs',
            'a rectangular table surface',
            'a metal table'
        ],
        'car': [
            'a red sports car',
            'a blue sedan vehicle',
            'a white family car',
            'a black luxury car',
            'a silver automobile'
        ],
        'airplane': [
            'a commercial passenger airplane',
            'a small private jet',
            'a white aircraft with wings',
            'a flying vehicle',
            'a modern airplane design'
        ]
    }
    
    # Generate point clouds for each category
    all_data = []
    
    for category, descriptions in categories.items():
        print(f"  Generating data for {category}...")
        
        for i, description in enumerate(descriptions):
            # Generate realistic point cloud for this category
            point_cloud = generate_category_point_cloud(category, i)
            
            all_data.append({
                'text': description,
                'point_cloud': point_cloud.tolist(),
                'category': category
            })
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
    
    # Save data
    data_dir = Path("data/comprehensive_training")
    data_dir.mkdir(exist_ok=True)
    
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_file = data_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f)
        print(f"  Saved {len(split_data)} samples to {split_name} split")
    
    return data_dir

def generate_category_point_cloud(category: str, variation: int) -> np.ndarray:
    """Generate a realistic point cloud for a specific category."""
    np.random.seed(variation)  # Ensure reproducibility
    
    if category == 'chair':
        return generate_chair_point_cloud()
    elif category == 'table':
        return generate_table_point_cloud()
    elif category == 'car':
        return generate_car_point_cloud()
    elif category == 'airplane':
        return generate_airplane_point_cloud()
    else:
        return np.random.randn(1024, 3) * 0.5

def generate_chair_point_cloud() -> np.ndarray:
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
    
    # Convert to numpy array and normalize
    points = np.array(points)
    if len(points) > 1024:
        indices = np.random.choice(len(points), 1024, replace=False)
        points = points[indices]
    else:
        padding = np.random.randn(1024 - len(points), 3) * 0.05
        points = np.vstack([points, padding])
    
    return points

def generate_table_point_cloud() -> np.ndarray:
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
    
    # Convert to numpy array and normalize
    points = np.array(points)
    if len(points) > 1024:
        indices = np.random.choice(len(points), 1024, replace=False)
        points = points[indices]
    else:
        padding = np.random.randn(1024 - len(points), 3) * 0.05
        points = np.vstack([points, padding])
    
    return points

def generate_car_point_cloud() -> np.ndarray:
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
    
    # Convert to numpy array and normalize
    points = np.array(points)
    if len(points) > 1024:
        indices = np.random.choice(len(points), 1024, replace=False)
        points = points[indices]
    else:
        padding = np.random.randn(1024 - len(points), 3) * 0.05
        points = np.vstack([points, padding])
    
    return points

def generate_airplane_point_cloud() -> np.ndarray:
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
    
    # Convert to numpy array and normalize
    points = np.array(points)
    if len(points) > 1024:
        indices = np.random.choice(len(points), 1024, replace=False)
        points = points[indices]
    else:
        padding = np.random.randn(1024 - len(points), 3) * 0.05
        points = np.vstack([points, padding])
    
    return points

# Example usage
def test_comprehensive_training():
    """Test the comprehensive training system."""
    print("Testing Comprehensive Training System...")
    
    # Create training data
    data_dir = create_training_data()
    
    # Load datasets
    train_data = json.load(open(data_dir / "train.json"))
    val_data = json.load(open(data_dir / "val.json"))
    
    train_dataset = TrainingDataset(train_data)
    val_dataset = TrainingDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Create model
    text_encoder = ImprovedTextEncoder()
    model = TextToPointCloudModel(text_encoder)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ComprehensiveTrainer(model, device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=10, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model("data/comprehensive_trained_model.pth")

if __name__ == "__main__":
    test_comprehensive_training()
