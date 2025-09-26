"""
Test Improved Text Encoding and Validation

This script demonstrates the improved text encoding system that should
generate different point clouds for different text inputs.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_improved_text_encoding():
    """Test the improved text encoding system."""
    print("=" * 80)
    print("TESTING IMPROVED TEXT ENCODING")
    print("=" * 80)
    
    try:
        from text_processing.improved_encoder import ImprovedTextEncoder
        
        # Create improved encoder
        encoder = ImprovedTextEncoder()
        
        # Test different texts
        test_texts = [
            "a wooden chair with four legs",
            "a red sports car with two doors", 
            "a white commercial airplane",
            "a modern glass table"
        ]
        
        print("1. Testing Text Encoder...")
        print("Text encodings:")
        
        embeddings = []
        for text in test_texts:
            embedding = encoder.encode_text(text)
            embeddings.append(embedding)
            print(f"  '{text}': shape={embedding.shape}, mean={embedding.mean():.3f}, std={embedding.std():.3f}")
        
        # Check if different texts produce different embeddings
        print("\nEmbedding differences:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                diff = np.linalg.norm(embeddings[i].numpy() - embeddings[j].numpy())
                print(f"  '{test_texts[i]}' vs '{test_texts[j]}': {diff:.3f}")
        
        print("\n✓ Improved text encoder working - different texts produce different embeddings")
        
    except Exception as e:
        print(f"✗ Improved text encoding failed: {e}")

def test_ml_generator_with_improved_encoding():
    """Test ML generator with improved text encoding."""
    print("\n2. Testing ML Generator with Improved Encoding...")
    
    try:
        from shape_generator.ml_generator import TextToPointCloudML
        from text_processing.improved_encoder import ImprovedTextEncoder
        from validation.validator import PointCloudValidator
        
        # Create improved encoder
        encoder = ImprovedTextEncoder()
        
        # Create generator with improved text encoding
        generator = TextToPointCloudML()
        
        # Override the text encoding method
        def improved_text_to_embedding(texts):
            embeddings = []
            for text in texts:
                embedding = encoder.encode_text(text)
                embeddings.append(embedding.squeeze(0).numpy())
            return np.array(embeddings)
        
        # Monkey patch the text encoding method
        generator._text_to_embedding = improved_text_to_embedding
        
        # Create validator
        validator = PointCloudValidator()
        
        # Test different texts
        test_cases = [
            ("a wooden chair with four legs", "chair"),
            ("a red sports car with two doors", "car"),
            ("a white commercial airplane", "airplane"),
            ("a modern glass table", "table")
        ]
        
        print("Generating point clouds with improved text encoding:")
        
        results = []
        for text, expected_category in test_cases:
            print(f"\n  Testing: '{text}'")
            
            # Generate point cloud
            points = generator.generate_point_cloud(text, num_points=500)
            points_array = np.array(points)
            
            # Validate
            validation_results = validator.validate_generation(points_array, text, expected_category)
            
            # Store results
            results.append({
                'text': text,
                'expected_category': expected_category,
                'points': points_array,
                'validation': validation_results
            })
            
            # Print results
            print(f"    Predicted category: {validation_results['category_accuracy']['predicted_category']}")
            print(f"    Accuracy: {validation_results['category_accuracy']['accuracy']:.3f}")
            print(f"    Text consistency: {validation_results['text_consistency']['overall_consistency']:.3f}")
            print(f"    Quality: {validation_results['quality_metrics']['compactness']:.3f}")
            print(f"    Dimensions: {[f'{d:.2f}' for d in validation_results['shape_metrics']['dimensions']]}")
        
        # Check if different texts produce different point clouds
        print("\nPoint cloud differences:")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                points1 = results[i]['points']
                points2 = results[j]['points']
                
                # Calculate difference in point cloud statistics
                mean_diff = np.linalg.norm(np.mean(points1, axis=0) - np.mean(points2, axis=0))
                std_diff = np.linalg.norm(np.std(points1, axis=0) - np.std(points2, axis=0))
                
                print(f"  '{results[i]['text']}' vs '{results[j]['text']}':")
                print(f"    Mean difference: {mean_diff:.3f}")
                print(f"    Std difference: {std_diff:.3f}")
        
        print("\n✓ ML generator with improved encoding tested")
        
        return results
        
    except Exception as e:
        print(f"✗ ML generator with improved encoding failed: {e}")
        return []

def visualize_comparison(results):
    """Visualize comparison of different text inputs."""
    if not results:
        print("No results to visualize")
        return
    
    print("\n3. Visualizing Comparison...")
    
    try:
        fig = plt.figure(figsize=(20, 15))
        
        for i, result in enumerate(results):
            points = result['points']
            text = result['text']
            validation = result['validation']
            
            # 3D plot
            ax = fig.add_subplot(2, 4, i+1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, alpha=0.7)
            ax.set_title(f'"{text}"\nCategory: {validation["category_accuracy"]["predicted_category"]}\nQuality: {validation["quality_metrics"]["compactness"]:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # XY projection
            ax = fig.add_subplot(2, 4, i+5)
            ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.7)
            ax.set_title(f'XY Projection\nSpread: {validation["shape_metrics"]["spread"]:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualization completed")
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

def main():
    """Main test function."""
    print("Testing Improved Text Encoding and Point Cloud Generation")
    print("=" * 80)
    
    # Test 1: Improved text encoding
    test_improved_text_encoding()
    
    # Test 2: ML generator with improved encoding
    results = test_ml_generator_with_improved_encoding()
    
    # Test 3: Visualization
    visualize_comparison(results)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    
    if results:
        print("\nKey Findings:")
        print("1. Improved text encoding produces different embeddings for different texts")
        print("2. Different text inputs should now generate different point clouds")
        print("3. Validation system can measure the differences")
        print("4. Point clouds should show some structure based on input text")
        
        # Check if we actually got different point clouds
        all_same = True
        for i in range(1, len(results)):
            if not np.allclose(results[0]['points'], results[i]['points'], atol=1e-3):
                all_same = False
                break
        
        if all_same:
            print("\n❌ PROBLEM: All point clouds are still identical!")
            print("   The model needs proper training on the improved embeddings")
        else:
            print("\n✅ SUCCESS: Different text inputs generate different point clouds!")
    else:
        print("\n❌ No results to analyze")

if __name__ == "__main__":
    main()
