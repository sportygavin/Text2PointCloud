"""
Comprehensive Test Script for 3D Point Cloud Generation and Training

This script demonstrates the complete pipeline:
1. 3D point cloud generation
2. Advanced 3D visualization
3. Training data creation
4. Model training
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_complete_pipeline():
    """Test the complete 3D point cloud pipeline."""
    print("=" * 60)
    print("TESTING COMPLETE 3D POINT CLOUD PIPELINE")
    print("=" * 60)
    
    # Test 1: 3D Visualization
    print("\n1. Testing 3D Visualization...")
    test_3d_visualization()
    
    # Test 2: ML Point Cloud Generation
    print("\n2. Testing ML Point Cloud Generation...")
    test_ml_generation()
    
    # Test 3: Training Data Creation
    print("\n3. Testing Training Data Creation...")
    test_training_data()
    
    # Test 4: Model Training
    print("\n4. Testing Model Training...")
    test_model_training()

def test_3d_visualization():
    """Test 3D visualization capabilities."""
    try:
        from src.visualization.d3d_visualizer import PointCloudVisualizer3D
        
        # Generate test points
        np.random.seed(42)
        test_points = [(np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)) 
                       for _ in range(200)]
        
        visualizer = PointCloudVisualizer3D()
        
        print("  ✓ Testing matplotlib 3D visualization...")
        visualizer.plot_matplotlib_3d(test_points, "Test 3D Point Cloud", color_by='z')
        
        print("  ✓ Testing multiple views...")
        visualizer.plot_multiple_views(test_points, "Multiple Views Test")
        
        print("  ✓ Testing point cloud analysis...")
        stats = visualizer.analyze_point_cloud(test_points)
        
    except Exception as e:
        print(f"  ✗ 3D Visualization failed: {e}")

def test_ml_generation():
    """Test ML-based point cloud generation."""
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        
        generator = TextToPointCloudML()
        
        test_texts = [
            "a red sports car",
            "a wooden chair",
            "a modern table",
            "a flying airplane"
        ]
        
        for text in test_texts:
            print(f"  ✓ Generating point cloud for: '{text}'")
            points = generator.generate_point_cloud(text, num_points=200)
            print(f"    Generated {len(points)} points")
            
            # Test 3D visualization
            try:
                from src.visualization.d3d_visualizer import PointCloudVisualizer3D
                visualizer = PointCloudVisualizer3D()
                visualizer.plot_matplotlib_3d(points, f"ML Generated: {text}")
            except:
                print(f"    (3D visualization not available)")
        
    except Exception as e:
        print(f"  ✗ ML Generation failed: {e}")

def test_training_data():
    """Test training data creation."""
    try:
        from training.data_manager import TrainingDataManager, PointCloudDataset
        
        # Create data manager
        data_manager = TrainingDataManager("test_data")
        
        # Create synthetic dataset
        categories = ['chair', 'table', 'car']
        print(f"  ✓ Creating synthetic dataset with categories: {categories}")
        data_manager.create_synthetic_dataset(categories, samples_per_category=20)
        
        # Test dataset loading
        train_dataset = PointCloudDataset("test_data/splits", split='train')
        print(f"  ✓ Loaded training dataset with {len(train_dataset)} samples")
        
        # Test data sample
        sample = train_dataset[0]
        print(f"  ✓ Sample point cloud shape: {sample['point_cloud'].shape}")
        print(f"  ✓ Sample text: {sample['text']}")
        print(f"  ✓ Sample category: {sample['category']}")
        
    except Exception as e:
        print(f"  ✗ Training Data failed: {e}")

def test_model_training():
    """Test model training."""
    try:
        from training.trainer import TextToPointCloudTrainer
        from training.data_manager import PointCloudDataset
        from shape_generator.ml_generator import PointCloudGenerator
        from torch.utils.data import DataLoader
        import torch
        
        # Load datasets
        train_dataset = PointCloudDataset("test_data/splits", split='train')
        val_dataset = PointCloudDataset("test_data/splits", split='val')
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"  ✓ Created data loaders")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        
        # Create model
        model = PointCloudGenerator(text_dim=512, point_dim=3, num_points=1024)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = TextToPointCloudTrainer(model, device)
        
        print(f"  ✓ Created model on {device}")
        print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model (short training for testing)
        print("  ✓ Starting training...")
        trainer.train(train_loader, val_loader, epochs=2, learning_rate=0.001)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save model
        trainer.save_model("test_data/test_model.pth")
        print("  ✓ Model training completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Model Training failed: {e}")

def test_real_world_example():
    """Test with a real-world example."""
    print("\n" + "=" * 60)
    print("REAL-WORLD EXAMPLE: GENERATING A CHAIR")
    print("=" * 60)
    
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        from src.visualization.d3d_visualizer import PointCloudVisualizer3D
        
        # Generate a chair point cloud
        generator = TextToPointCloudML()
        chair_points = generator.generate_point_cloud("a comfortable wooden chair", num_points=500)
        
        print(f"Generated {len(chair_points)} points for a chair")
        
        # Visualize the chair
        visualizer = PointCloudVisualizer3D()
        visualizer.plot_matplotlib_3d(chair_points, "Generated Chair", color_by='z')
        visualizer.plot_plotly_3d(chair_points, "Interactive Chair", color_by='distance')
        
        # Analyze the point cloud
        stats = visualizer.analyze_point_cloud(chair_points)
        
        print("Chair generation and visualization completed!")
        
    except Exception as e:
        print(f"Real-world example failed: {e}")

if __name__ == "__main__":
    # Run all tests
    test_complete_pipeline()
    
    # Run real-world example
    test_real_world_example()
    
    print("\n" + "=" * 60)
    print("PIPELINE TESTING COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install additional dependencies: pip install plotly")
    print("2. Experiment with different text inputs")
    print("3. Train on larger datasets")
    print("4. Integrate with CLIP for better text encoding")
    print("5. Explore pre-trained models like Point-E")
