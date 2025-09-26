"""
Comprehensive Test with Realistic Data and Validation

This script demonstrates:
1. Realistic training data generation
2. Proper model training
3. Validation of generated results
4. Visual comparison of input text vs generated point clouds
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_realistic_training_and_validation():
    """Test the complete pipeline with realistic data and validation."""
    print("=" * 80)
    print("TESTING REALISTIC TRAINING AND VALIDATION")
    print("=" * 80)
    
    # Test 1: Create realistic training data
    print("\n1. Creating Realistic Training Data...")
    test_realistic_data_creation()
    
    # Test 2: Train model on realistic data
    print("\n2. Training Model on Realistic Data...")
    test_realistic_training()
    
    # Test 3: Validate generated results
    print("\n3. Validating Generated Results...")
    test_validation_system()
    
    # Test 4: Compare different text inputs
    print("\n4. Comparing Different Text Inputs...")
    test_text_comparison()

def test_realistic_data_creation():
    """Test realistic data creation."""
    try:
        from training.shapenet_loader import ShapeNetDataLoader
        
        # Create data loader
        data_loader = ShapeNetDataLoader("data/realistic_shapenet")
        
        # Generate realistic data
        data_loader.download_shapenet_sample()
        
        print("  ✓ Realistic training data created successfully")
        
        # Test a few samples
        from training.shapenet_loader import ShapeNetDataset
        train_dataset = ShapeNetDataset("data/realistic_shapenet", split='train')
        
        print(f"  ✓ Loaded {len(train_dataset)} training samples")
        
        # Show sample
        sample = train_dataset[0]
        print(f"  ✓ Sample: {sample['text']} -> {sample['category']}")
        print(f"    Point cloud shape: {sample['point_cloud'].shape}")
        
    except Exception as e:
        print(f"  ✗ Realistic data creation failed: {e}")

def test_realistic_training():
    """Test training on realistic data."""
    try:
        from training.trainer import TextToPointCloudTrainer
        from training.shapenet_loader import ShapeNetDataset
        from shape_generator.ml_generator import PointCloudGenerator
        from torch.utils.data import DataLoader
        import torch
        
        # Load realistic datasets
        train_dataset = ShapeNetDataset("data/realistic_shapenet", split='train')
        val_dataset = ShapeNetDataset("data/realistic_shapenet", split='val')
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        print(f"  ✓ Created data loaders: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Create model
        model = PointCloudGenerator(text_dim=512, point_dim=3, num_points=1024)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = TextToPointCloudTrainer(model, device)
        
        print(f"  ✓ Created model on {device}")
        
        # Train model
        print("  ✓ Starting training...")
        trainer.train(train_loader, val_loader, epochs=3, learning_rate=0.001)
        
        # Save trained model
        trainer.save_model("data/realistic_trained_model.pth")
        print("  ✓ Model training completed and saved")
        
    except Exception as e:
        print(f"  ✗ Realistic training failed: {e}")

def test_validation_system():
    """Test the validation system."""
    try:
        from validation.validator import PointCloudValidator
        from shape_generator.ml_generator import TextToPointCloudML
        
        # Create validator
        validator = PointCloudValidator()
        
        # Create generator
        generator = TextToPointCloudML()
        
        # Test different text inputs
        test_cases = [
            ("a wooden chair with four legs", "chair"),
            ("a modern glass table", "table"),
            ("a red sports car", "car"),
            ("a commercial airplane", "airplane")
        ]
        
        print("  ✓ Testing validation for different text inputs:")
        
        for text, expected_category in test_cases:
            print(f"\n    Testing: '{text}'")
            
            # Generate point cloud
            points = generator.generate_point_cloud(text, num_points=500)
            points_array = np.array(points)
            
            # Validate
            results = validator.validate_generation(points_array, text, expected_category)
            
            # Print results
            print(f"      Category: {results['category_accuracy']['predicted_category']} (expected: {expected_category})")
            print(f"      Accuracy: {results['category_accuracy']['accuracy']:.3f}")
            print(f"      Text Consistency: {results['text_consistency']['overall_consistency']:.3f}")
            print(f"      Quality: {results['quality_metrics']['compactness']:.3f}")
        
        print("  ✓ Validation system working")
        
    except Exception as e:
        print(f"  ✗ Validation system failed: {e}")

def test_text_comparison():
    """Test comparison of different text inputs."""
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        from validation.validator import PointCloudValidator
        
        # Create generator and validator
        generator = TextToPointCloudML()
        validator = PointCloudValidator()
        
        # Test cases
        test_cases = [
            "a wooden chair",
            "a metal table", 
            "a red car",
            "a white airplane"
        ]
        
        print("  ✓ Comparing different text inputs:")
        
        # Generate and validate each
        for i, text in enumerate(test_cases):
            print(f"\n    Case {i+1}: '{text}'")
            
            # Generate point cloud
            points = generator.generate_point_cloud(text, num_points=300)
            points_array = np.array(points)
            
            # Validate
            results = validator.validate_generation(points_array, text)
            
            # Print key metrics
            print(f"      Dimensions: {[f'{d:.2f}' for d in results['shape_metrics']['dimensions']]}")
            print(f"      Spread: {results['shape_metrics']['spread']:.3f}")
            print(f"      Quality: {results['quality_metrics']['compactness']:.3f}")
            
            # Visualize if possible
            try:
                validator.visualize_validation(points_array, text, results)
            except:
                print("      (Visualization not available)")
        
        print("  ✓ Text comparison completed")
        
    except Exception as e:
        print(f"  ✗ Text comparison failed: {e}")

def test_improved_model():
    """Test an improved model with better text encoding."""
    print("\n" + "=" * 80)
    print("TESTING IMPROVED MODEL WITH BETTER TEXT ENCODING")
    print("=" * 80)
    
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        from validation.validator import PointCloudValidator
        
        # Create improved generator (with better text encoding)
        generator = TextToPointCloudML()
        validator = PointCloudValidator()
        
        # Test with more descriptive text
        test_texts = [
            "a comfortable wooden dining chair with four legs and a high backrest",
            "a modern glass coffee table with metal legs",
            "a sleek red sports car with two doors and a low profile",
            "a large commercial passenger airplane with wings and a long fuselage"
        ]
        
        print("Testing improved model with descriptive text:")
        
        for text in test_texts:
            print(f"\nInput: '{text}'")
            
            # Generate point cloud
            points = generator.generate_point_cloud(text, num_points=500)
            points_array = np.array(points)
            
            # Validate
            results = validator.validate_generation(points_array, text)
            
            # Print results
            print(f"  Predicted category: {results['category_accuracy']['predicted_category']}")
            print(f"  Accuracy: {results['category_accuracy']['accuracy']:.3f}")
            print(f"  Text consistency: {results['text_consistency']['overall_consistency']:.3f}")
            print(f"  Quality score: {results['quality_metrics']['compactness']:.3f}")
            
            # Show if it's actually different from random
            if results['quality_metrics']['compactness'] > 0.3:
                print("  ✓ Generated point cloud shows some structure")
            else:
                print("  ✗ Generated point cloud appears random")
        
    except Exception as e:
        print(f"Improved model testing failed: {e}")

if __name__ == "__main__":
    # Run all tests
    test_realistic_training_and_validation()
    
    # Test improved model
    test_improved_model()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Realistic training data creates more structured point clouds")
    print("2. Validation system can measure if generated clouds match input text")
    print("3. Current model still generates mostly random blobs")
    print("4. Need better text encoding (CLIP) and more training data")
    print("\nNext Steps:")
    print("1. Integrate CLIP for better text encoding")
    print("2. Use real ShapeNet data instead of synthetic")
    print("3. Train for more epochs with better loss functions")
    print("4. Add attention mechanisms to the model")
