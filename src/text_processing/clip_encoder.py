"""
Text2PointCloud - CLIP Integration

This module integrates CLIP for state-of-the-art text encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import clip
import warnings
warnings.filterwarnings("ignore")

class CLIPTextEncoder:
    """
    CLIP-based text encoder for semantic text understanding.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model name (ViT-B/32, ViT-L/14, etc.)
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.available = True
            print(f"✓ CLIP loaded successfully on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load CLIP: {e}")
            print("Installing CLIP...")
            self._install_clip()
            try:
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                self.available = True
                print(f"✓ CLIP loaded successfully after installation")
            except Exception as e2:
                print(f"✗ CLIP installation failed: {e2}")
                self.available = False
        
        # Get text encoding dimension
        if self.available:
            self.text_dim = self.model.text_projection.shape[1]
            print(f"Text encoding dimension: {self.text_dim}")
        else:
            self.text_dim = 512  # Fallback dimension
    
    def _install_clip(self):
        """Install CLIP if not available."""
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "clip-by-openai"])
        except Exception as e:
            print(f"Failed to install CLIP: {e}")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text using CLIP.
        
        Args:
            text: Input text description
            
        Returns:
            CLIP text embedding tensor
        """
        if not self.available:
            # Fallback to simple encoding
            return self._fallback_encoding(text)
        
        try:
            # Tokenize text
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            
            # Encode with CLIP
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            return text_features
        except Exception as e:
            print(f"CLIP encoding failed: {e}, using fallback")
            return self._fallback_encoding(text)
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts using CLIP.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            CLIP text embeddings tensor
        """
        if not self.available:
            # Fallback to simple encoding
            return torch.stack([self._fallback_encoding(text) for text in texts])
        
        try:
            # Tokenize texts
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            
            # Encode with CLIP
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            return text_features
        except Exception as e:
            print(f"CLIP batch encoding failed: {e}, using fallback")
            return torch.stack([self._fallback_encoding(text) for text in texts])
    
    def _fallback_encoding(self, text: str) -> torch.Tensor:
        """Fallback encoding when CLIP is not available."""
        # Simple hash-based encoding as fallback
        text_hash = abs(hash(text.lower())) % (2**32 - 1)
        embedding = np.random.RandomState(text_hash).normal(0, 1, self.text_dim)
        return torch.tensor(embedding, dtype=torch.float32)
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Get similarity between two texts using CLIP.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.available:
            return 0.5  # Neutral similarity for fallback
        
        try:
            # Encode both texts
            emb1 = self.encode_text(text1)
            emb2 = self.encode_text(text2)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2, dim=1).item()
            return similarity
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.5
    
    def get_text_features(self, text: str) -> Dict:
        """
        Get detailed text features using CLIP.
        
        Args:
            text: Input text description
            
        Returns:
            Dictionary with text features
        """
        if not self.available:
            return {"embedding": self._fallback_encoding(text), "confidence": 0.5}
        
        try:
            # Encode text
            embedding = self.encode_text(text)
            
            # Calculate confidence based on embedding norm
            confidence = torch.norm(embedding).item()
            
            # Extract keywords (simple approach)
            keywords = self._extract_keywords(text)
            
            return {
                "embedding": embedding,
                "confidence": confidence,
                "keywords": keywords,
                "text_length": len(text),
                "word_count": len(text.split())
            }
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return {"embedding": self._fallback_encoding(text), "confidence": 0.5}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        text_lower = text.lower()
        keywords = []
        
        # Object categories
        categories = ['chair', 'table', 'car', 'airplane', 'lamp', 'sofa', 'bed', 'bottle', 'bowl']
        for category in categories:
            if category in text_lower:
                keywords.append(category)
        
        # Materials
        materials = ['wooden', 'metal', 'plastic', 'leather', 'glass', 'fabric']
        for material in materials:
            if material in text_lower:
                keywords.append(material)
        
        # Colors
        colors = ['red', 'blue', 'white', 'black', 'green', 'yellow', 'brown', 'silver']
        for color in colors:
            if color in text_lower:
                keywords.append(color)
        
        return keywords

class CLIPTextToPointCloudModel(nn.Module):
    """
    Text-to-point-cloud model using CLIP for text encoding.
    """
    
    def __init__(self, clip_encoder: CLIPTextEncoder, point_dim: int = 3, num_points: int = 1024):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.point_dim = point_dim
        self.num_points = num_points
        self.text_dim = clip_encoder.text_dim
        
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
        Generate point clouds from text inputs using CLIP.
        
        Args:
            text_inputs: List of text descriptions
            
        Returns:
            Point cloud tensor of shape (batch_size, num_points, point_dim)
        """
        # Encode text with CLIP
        text_embeddings = self.clip_encoder.encode_texts(text_inputs)
        
        # Generate point clouds
        point_clouds = self.point_decoder(text_embeddings)
        
        # Reshape to point cloud format
        point_clouds = point_clouds.view(-1, self.num_points, self.point_dim)
        
        return point_clouds

class CLIPTextToPointCloudGenerator:
    """
    Main generator class using CLIP for text encoding.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """Initialize the CLIP-based generator."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create CLIP encoder
        self.clip_encoder = CLIPTextEncoder(model_name, self.device)
        
        # Create model
        self.model = CLIPTextToPointCloudModel(self.clip_encoder).to(self.device)
        
        print(f"CLIP-based generator initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate_point_cloud(self, text: str, num_points: int = 1024) -> List[Tuple[float, float, float]]:
        """
        Generate point cloud from text using CLIP.
        
        Args:
            text: Text description
            num_points: Number of points in generated cloud
            
        Returns:
            List of (x, y, z) coordinate tuples
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate point cloud
            point_cloud = self.model([text])
            point_cloud = point_cloud.squeeze(0).cpu().numpy()
        
        # Convert to list of tuples
        points = [(float(x), float(y), float(z)) for x, y, z in point_cloud]
        
        return points
    
    def generate_multiple_point_clouds(self, texts: List[str], num_points: int = 1024) -> List[List[Tuple[float, float, float]]]:
        """
        Generate multiple point clouds from texts using CLIP.
        
        Args:
            texts: List of text descriptions
            num_points: Number of points in each generated cloud
            
        Returns:
            List of point cloud lists
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate point clouds
            point_clouds = self.model(texts)
            point_clouds = point_clouds.cpu().numpy()
        
        # Convert to list of tuples
        results = []
        for point_cloud in point_clouds:
            points = [(float(x), float(y), float(z)) for x, y, z in point_cloud]
            results.append(points)
        
        return results
    
    def get_text_similarity(self, text1: str, text2: str) -> float:
        """Get similarity between two texts using CLIP."""
        return self.clip_encoder.get_similarity(text1, text2)
    
    def get_text_features(self, text: str) -> Dict:
        """Get detailed text features using CLIP."""
        return self.clip_encoder.get_text_features(text)

# Test functions
def test_clip_integration():
    """Test CLIP integration."""
    print("Testing CLIP Integration...")
    
    try:
        # Create CLIP generator
        generator = CLIPTextToPointCloudGenerator()
        
        # Test different texts
        test_texts = [
            "a wooden chair with four legs",
            "a red sports car with two doors",
            "a white commercial airplane",
            "a modern glass table"
        ]
        
        print("\nTesting text similarity:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                similarity = generator.get_text_similarity(test_texts[i], test_texts[j])
                print(f"  '{test_texts[i]}' vs '{test_texts[j]}': {similarity:.3f}")
        
        print("\nTesting point cloud generation:")
        for text in test_texts:
            print(f"  Generating point cloud for: '{text}'")
            points = generator.generate_point_cloud(text, num_points=100)
            print(f"    Generated {len(points)} points")
            print(f"    First 3 points: {points[:3]}")
        
        print("\n✓ CLIP integration working!")
        
    except Exception as e:
        print(f"✗ CLIP integration failed: {e}")

if __name__ == "__main__":
    test_clip_integration()
