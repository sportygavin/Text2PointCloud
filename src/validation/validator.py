"""
Text2PointCloud - Validation and Metrics System

This module provides validation metrics to measure if generated point clouds
actually match the input text descriptions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class PointCloudValidator:
    """
    Validates generated point clouds against input text descriptions.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.category_classifier = None
        self.text_encoder = None
    
    def validate_generation(self, generated_points: np.ndarray, 
                          input_text: str, expected_category: str = None) -> Dict:
        """
        Validate if generated point cloud matches input text.
        
        Args:
            generated_points: Generated point cloud (N, 3)
            input_text: Input text description
            expected_category: Expected object category
            
        Returns:
            Dictionary with validation metrics
        """
        results = {}
        
        # 1. Basic shape analysis
        results['shape_metrics'] = self._analyze_shape(generated_points)
        
        # 2. Category classification
        if expected_category:
            results['category_accuracy'] = self._classify_category(generated_points, expected_category)
        
        # 3. Text consistency
        results['text_consistency'] = self._check_text_consistency(generated_points, input_text)
        
        # 4. Quality metrics
        results['quality_metrics'] = self._calculate_quality_metrics(generated_points)
        
        return results
    
    def _analyze_shape(self, points: np.ndarray) -> Dict:
        """Analyze basic shape properties of the point cloud."""
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate center
        center = np.mean(points, axis=0)
        
        # Calculate spread
        distances = np.linalg.norm(points - center, axis=1)
        spread = np.std(distances)
        
        # Calculate aspect ratios
        aspect_ratios = dimensions / np.max(dimensions)
        
        return {
            'dimensions': dimensions.tolist(),
            'center': center.tolist(),
            'spread': float(spread),
            'aspect_ratios': aspect_ratios.tolist(),
            'num_points': len(points)
        }
    
    def _classify_category(self, points: np.ndarray, expected_category: str) -> Dict:
        """Classify the point cloud category and compare with expected."""
        # Simple rule-based classification based on shape analysis
        shape_metrics = self._analyze_shape(points)
        dimensions = shape_metrics['dimensions']
        aspect_ratios = shape_metrics['aspect_ratios']
        
        # Classification rules
        predicted_category = self._rule_based_classification(dimensions, aspect_ratios)
        
        # Calculate accuracy
        accuracy = 1.0 if predicted_category == expected_category else 0.0
        
        return {
            'predicted_category': predicted_category,
            'expected_category': expected_category,
            'accuracy': accuracy,
            'confidence': self._calculate_confidence(points, predicted_category)
        }
    
    def _rule_based_classification(self, dimensions: List[float], aspect_ratios: List[float]) -> str:
        """Simple rule-based classification of point clouds."""
        x_dim, y_dim, z_dim = dimensions
        x_ratio, y_ratio, z_ratio = aspect_ratios
        
        # Chair: roughly square in XY, taller in Z
        if z_ratio > 0.7 and x_ratio > 0.5 and y_ratio > 0.5:
            return 'chair'
        
        # Table: flat in Z, square in XY
        elif z_ratio < 0.3 and x_ratio > 0.7 and y_ratio > 0.7:
            return 'table'
        
        # Car: elongated in X, flat in Z
        elif x_ratio > 0.8 and z_ratio < 0.4:
            return 'car'
        
        # Airplane: very elongated in X, flat in Z
        elif x_ratio > 0.9 and z_ratio < 0.3:
            return 'airplane'
        
        # Default
        else:
            return 'unknown'
    
    def _calculate_confidence(self, points: np.ndarray, category: str) -> float:
        """Calculate confidence score for category prediction."""
        # Simple confidence based on how well the shape matches expected patterns
        shape_metrics = self._analyze_shape(points)
        dimensions = shape_metrics['dimensions']
        aspect_ratios = shape_metrics['aspect_ratios']
        
        # Define expected patterns for each category
        expected_patterns = {
            'chair': {'x_ratio': 0.6, 'y_ratio': 0.6, 'z_ratio': 0.8},
            'table': {'x_ratio': 0.8, 'y_ratio': 0.8, 'z_ratio': 0.2},
            'car': {'x_ratio': 0.9, 'y_ratio': 0.4, 'z_ratio': 0.3},
            'airplane': {'x_ratio': 0.95, 'y_ratio': 0.3, 'z_ratio': 0.2}
        }
        
        if category not in expected_patterns:
            return 0.5
        
        expected = expected_patterns[category]
        actual = {
            'x_ratio': aspect_ratios[0],
            'y_ratio': aspect_ratios[1],
            'z_ratio': aspect_ratios[2]
        }
        
        # Calculate similarity
        similarity = 1.0 - np.mean([abs(actual[key] - expected[key]) for key in expected])
        return max(0.0, min(1.0, similarity))
    
    def _check_text_consistency(self, points: np.ndarray, text: str) -> Dict:
        """Check if point cloud is consistent with text description."""
        # Extract keywords from text
        keywords = self._extract_keywords(text)
        
        # Check consistency
        consistency_scores = {}
        
        for keyword in keywords:
            if keyword in ['chair', 'table', 'car', 'airplane']:
                # Check if shape matches category
                shape_metrics = self._analyze_shape(points)
                predicted_category = self._rule_based_classification(
                    shape_metrics['dimensions'], 
                    shape_metrics['aspect_ratios']
                )
                consistency_scores[keyword] = 1.0 if predicted_category == keyword else 0.0
            
            elif keyword in ['wooden', 'metal', 'plastic']:
                # Material keywords - can't really validate from point cloud
                consistency_scores[keyword] = 0.5  # Neutral score
            
            elif keyword in ['red', 'blue', 'white', 'black']:
                # Color keywords - can't validate from point cloud
                consistency_scores[keyword] = 0.5  # Neutral score
            
            else:
                # Other keywords
                consistency_scores[keyword] = 0.5  # Neutral score
        
        return {
            'keywords': keywords,
            'scores': consistency_scores,
            'overall_consistency': np.mean(list(consistency_scores.values()))
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text description."""
        # Simple keyword extraction
        text_lower = text.lower()
        keywords = []
        
        # Object categories
        categories = ['chair', 'table', 'car', 'airplane', 'lamp', 'sofa', 'bed']
        for category in categories:
            if category in text_lower:
                keywords.append(category)
        
        # Materials
        materials = ['wooden', 'metal', 'plastic', 'leather', 'glass']
        for material in materials:
            if material in text_lower:
                keywords.append(material)
        
        # Colors
        colors = ['red', 'blue', 'white', 'black', 'green', 'yellow']
        for color in colors:
            if color in text_lower:
                keywords.append(color)
        
        # Other descriptive words
        descriptors = ['small', 'large', 'modern', 'traditional', 'comfortable']
        for descriptor in descriptors:
            if descriptor in text_lower:
                keywords.append(descriptor)
        
        return keywords
    
    def _calculate_quality_metrics(self, points: np.ndarray) -> Dict:
        """Calculate quality metrics for the point cloud."""
        # Point density
        bounding_box_volume = np.prod(np.max(points, axis=0) - np.min(points, axis=0))
        point_density = len(points) / max(bounding_box_volume, 1e-6)
        
        # Point distribution uniformity
        distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
        distribution_uniformity = 1.0 / (1.0 + np.std(distances))
        
        # Compactness
        center = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - center, axis=1)
        compactness = 1.0 / (1.0 + np.mean(distances_to_center))
        
        return {
            'point_density': float(point_density),
            'distribution_uniformity': float(distribution_uniformity),
            'compactness': float(compactness),
            'num_points': len(points)
        }
    
    def visualize_validation(self, generated_points: np.ndarray, 
                           input_text: str, validation_results: Dict):
        """Visualize validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 3D point cloud
        ax1 = axes[0, 0]
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(generated_points[:, 0], generated_points[:, 1], generated_points[:, 2], 
                   s=20, alpha=0.7)
        ax1.set_title(f'Generated Point Cloud\n"{input_text}"')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 2. XY projection
        ax2 = axes[0, 1]
        ax2.scatter(generated_points[:, 0], generated_points[:, 1], s=20, alpha=0.7)
        ax2.set_title('XY Projection')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        # 3. XZ projection
        ax3 = axes[1, 0]
        ax3.scatter(generated_points[:, 0], generated_points[:, 2], s=20, alpha=0.7)
        ax3.set_title('XZ Projection')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.grid(True, alpha=0.3)
        
        # 4. Validation metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Display validation results
        metrics_text = f"""
Validation Results:
==================

Shape Metrics:
- Dimensions: {validation_results['shape_metrics']['dimensions']}
- Spread: {validation_results['shape_metrics']['spread']:.3f}
- Points: {validation_results['shape_metrics']['num_points']}

Category Classification:
- Predicted: {validation_results.get('category_accuracy', {}).get('predicted_category', 'N/A')}
- Expected: {validation_results.get('category_accuracy', {}).get('expected_category', 'N/A')}
- Accuracy: {validation_results.get('category_accuracy', {}).get('accuracy', 0.0):.3f}

Text Consistency:
- Overall: {validation_results.get('text_consistency', {}).get('overall_consistency', 0.0):.3f}

Quality Metrics:
- Density: {validation_results['quality_metrics']['point_density']:.3f}
- Uniformity: {validation_results['quality_metrics']['distribution_uniformity']:.3f}
- Compactness: {validation_results['quality_metrics']['compactness']:.3f}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

# Example usage
def test_validation_system():
    """Test the validation system."""
    print("Testing Validation System...")
    
    # Create validator
    validator = PointCloudValidator()
    
    # Generate test point cloud
    np.random.seed(42)
    test_points = np.random.randn(1024, 3) * 0.5
    
    # Test validation
    results = validator.validate_generation(
        test_points, 
        "a wooden chair with four legs", 
        "chair"
    )
    
    print("Validation Results:")
    print(f"Shape metrics: {results['shape_metrics']}")
    print(f"Category accuracy: {results['category_accuracy']}")
    print(f"Text consistency: {results['text_consistency']}")
    print(f"Quality metrics: {results['quality_metrics']}")
    
    # Visualize results
    validator.visualize_validation(test_points, "a wooden chair", results)

if __name__ == "__main__":
    test_validation_system()
