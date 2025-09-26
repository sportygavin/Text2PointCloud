"""
Text2PointCloud - Command Line Interface

A simple command line interface for text-to-point-cloud generation.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.improved_encoder import ImprovedTextEncoder
from validation.validator import PointCloudValidator
import torch

def load_model():
    """Load the trained model."""
    try:
        from test_comprehensive_shapenet_training import TextToPointCloudModel
        
        # Create model
        text_encoder = ImprovedTextEncoder()
        model = TextToPointCloudModel(text_encoder)
        
        # Load trained weights
        model_path = "data/comprehensive_trained_model.pth"
        if os.path.exists(model_path):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print("✓ Trained model loaded successfully")
            return model, text_encoder, device
        else:
            print("⚠️ Trained model not found, using untrained model")
            return model, text_encoder, 'cpu'
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None

def generate_point_cloud(model, text_encoder, device, text, num_points=1024):
    """Generate point cloud from text."""
    if model is None:
        print("Error: Model not loaded")
        return None, None
    
    try:
        # Generate point cloud
        model.eval()
        with torch.no_grad():
            point_cloud = model([text])
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
        
        # Convert to list of tuples
        points = [(float(x), float(y), float(z)) for x, y, z in point_cloud]
        
        # Validate
        validator = PointCloudValidator()
        validation_results = validator.validate_generation(np.array(points), text)
        
        return points, validation_results
        
    except Exception as e:
        print(f"Error generating point cloud: {e}")
        return None, None

def plot_point_cloud(points, text, save_path=None):
    """Plot the point cloud in 3D."""
    if not points:
        print("No points to plot")
        return
    
    # Extract coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=z_coords, cmap='viridis', s=20, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Generated Point Cloud: '{text}'")
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Add colorbar
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Z Coordinate')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def print_results(text, points, validation_results):
    """Print generation results."""
    print("\n" + "="*60)
    print(f"GENERATED POINT CLOUD FOR: '{text}'")
    print("="*60)
    
    print(f"Number of points: {len(points)}")
    
    if validation_results:
        print("\nVALIDATION RESULTS:")
        print("-" * 30)
        
        # Category accuracy
        if 'category_accuracy' in validation_results:
            cat_acc = validation_results['category_accuracy']
            print(f"Predicted Category: {cat_acc.get('predicted_category', 'Unknown')}")
            print(f"Confidence: {cat_acc.get('confidence', 0):.3f}")
        
        # Quality metrics
        if 'quality_metrics' in validation_results:
            quality = validation_results['quality_metrics']
            print(f"\nQuality Metrics:")
            print(f"  Compactness: {quality.get('compactness', 0):.3f}")
            print(f"  Spread: {quality.get('spread', 0):.3f}")
            print(f"  Density: {quality.get('density', 0):.3f}")
        
        # Shape metrics
        if 'shape_metrics' in validation_results:
            shape = validation_results['shape_metrics']
            print(f"\nShape Metrics:")
            dims = shape.get('dimensions', [0, 0, 0])
            print(f"  Dimensions: {dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f}")
            print(f"  Volume: {shape.get('volume', 0):.3f}")
            print(f"  Surface Area: {shape.get('surface_area', 0):.3f}")
        
        # Text consistency
        if 'text_consistency' in validation_results:
            text_cons = validation_results['text_consistency']
            print(f"\nText Consistency:")
            print(f"  Overall: {text_cons.get('overall_consistency', 0):.3f}")
            print(f"  Semantic: {text_cons.get('semantic_consistency', 0):.3f}")
            print(f"  Structural: {text_cons.get('structural_consistency', 0):.3f}")
    
    # Point cloud statistics
    if points:
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        print(f"\nPoint Cloud Statistics:")
        print(f"  X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"  Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"  Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
        print(f"  Center: ({np.mean(x_coords):.3f}, {np.mean(y_coords):.3f}, {np.mean(z_coords):.3f})")

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Text2PointCloud - Generate 3D Point Clouds from Text')
    parser.add_argument('text', help='Text description of the object to generate')
    parser.add_argument('--num-points', type=int, default=1024, help='Number of points to generate (default: 1024)')
    parser.add_argument('--save-plot', type=str, help='Save plot to file (e.g., output.png)')
    parser.add_argument('--no-plot', action='store_true', help='Do not display plot')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode - enter multiple texts')
    
    args = parser.parse_args()
    
    print("🚀 Text2PointCloud Command Line Interface")
    print("="*50)
    
    # Load model
    print("Loading model...")
    model, text_encoder, device = load_model()
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print(f"Model loaded on device: {device}")
    
    if args.interactive:
        # Interactive mode
        print("\n🔄 Interactive Mode - Enter text descriptions (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            try:
                text = input("\nEnter text description: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    print("Please enter some text")
                    continue
                
                # Generate point cloud
                points, validation_results = generate_point_cloud(
                    model, text_encoder, device, text, args.num_points
                )
                
                if points is not None:
                    # Print results
                    print_results(text, points, validation_results)
                    
                    # Plot if not disabled
                    if not args.no_plot:
                        plot_point_cloud(points, text, args.save_plot)
                else:
                    print("Failed to generate point cloud")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Single generation mode
        print(f"\nGenerating point cloud for: '{args.text}'")
        
        # Generate point cloud
        points, validation_results = generate_point_cloud(
            model, text_encoder, device, args.text, args.num_points
        )
        
        if points is not None:
            # Print results
            print_results(args.text, points, validation_results)
            
            # Plot if not disabled
            if not args.no_plot:
                plot_point_cloud(points, args.text, args.save_plot)
        else:
            print("Failed to generate point cloud")

if __name__ == "__main__":
    main()
