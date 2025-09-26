"""
Text2PointCloud - Improved Text Encoding

This module provides better text encoding for the ML model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
import re
from collections import Counter

class ImprovedTextEncoder:
    """
    Improved text encoder that captures semantic meaning.
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 512):
        """Initialize the text encoder."""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Create vocabulary from common words
        self.vocab = self._create_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Create embedding layer
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _create_vocabulary(self) -> List[str]:
        """Create vocabulary from common object-related words."""
        # Object categories
        categories = ['chair', 'table', 'car', 'airplane', 'lamp', 'sofa', 'bed', 'bottle', 'bowl']
        
        # Materials
        materials = ['wooden', 'metal', 'plastic', 'leather', 'glass', 'fabric']
        
        # Colors
        colors = ['red', 'blue', 'white', 'black', 'green', 'yellow', 'brown', 'silver']
        
        # Descriptive words
        descriptors = ['small', 'large', 'modern', 'traditional', 'comfortable', 'sleek', 'old', 'new']
        
        # Shape words
        shapes = ['round', 'square', 'rectangular', 'long', 'short', 'tall', 'wide', 'narrow']
        
        # Parts
        parts = ['legs', 'wings', 'doors', 'windows', 'wheels', 'backrest', 'armrests', 'seat']
        
        # Combine all words
        all_words = categories + materials + colors + descriptors + shapes + parts
        
        # Add common words
        common_words = ['a', 'an', 'the', 'with', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for']
        
        # Create vocabulary
        vocab = list(set(all_words + common_words))
        vocab.sort()
        
        # Add special tokens
        vocab = ['<PAD>', '<UNK>'] + vocab
        
        return vocab[:self.vocab_size]
    
    def _initialize_embeddings(self):
        """Initialize embeddings with semantic relationships."""
        with torch.no_grad():
            # Initialize with small random values
            self.embedding.weight.normal_(0, 0.1)
            
            # Create some semantic relationships
            self._create_semantic_relationships()
    
    def _create_semantic_relationships(self):
        """Create semantic relationships in embeddings."""
        with torch.no_grad():
            # Group related words
            categories = ['chair', 'table', 'car', 'airplane']
            materials = ['wooden', 'metal', 'plastic', 'glass']
            colors = ['red', 'blue', 'white', 'black']
            
            # Create category embeddings
            for i, category in enumerate(categories):
                if category in self.word_to_idx:
                    idx = self.word_to_idx[category]
                    # Create a unique embedding for each category
                    embedding = torch.zeros(self.embed_dim)
                    embedding[i] = 1.0  # One-hot encoding for category
                    embedding[i + 10] = 1.0  # Additional dimension
                    self.embedding.weight[idx] = embedding
            
            # Create material embeddings
            for i, material in enumerate(materials):
                if material in self.word_to_idx:
                    idx = self.word_to_idx[material]
                    embedding = torch.zeros(self.embed_dim)
                    embedding[20 + i] = 1.0  # Material dimension
                    self.embedding.weight[idx] = embedding
            
            # Create color embeddings
            for i, color in enumerate(colors):
                if color in self.word_to_idx:
                    idx = self.word_to_idx[color]
                    embedding = torch.zeros(self.embed_dim)
                    embedding[30 + i] = 1.0  # Color dimension
                    self.embedding.weight[idx] = embedding
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into a semantic embedding.
        
        Args:
            text: Input text description
            
        Returns:
            Text embedding tensor
        """
        # Tokenize text
        tokens = self._tokenize(text)
        
        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # Convert to tensor
        token_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Get embeddings
        embeddings = self.embedding(token_tensor)
        
        # Pool embeddings (mean pooling)
        pooled_embedding = torch.mean(embeddings, dim=0)
        
        # Add some text-specific features
        text_features = self._extract_text_features(text)
        text_features_tensor = torch.tensor(text_features, dtype=torch.float32)
        
        # Combine embeddings and features
        combined_embedding = torch.cat([pooled_embedding, text_features_tensor])
        
        # Ensure correct size
        if len(combined_embedding) > self.embed_dim:
            combined_embedding = combined_embedding[:self.embed_dim]
        elif len(combined_embedding) < self.embed_dim:
            padding = torch.zeros(self.embed_dim - len(combined_embedding))
            combined_embedding = torch.cat([combined_embedding, padding])
        
        return combined_embedding.unsqueeze(0)  # Add batch dimension
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        return words
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract additional features from text."""
        features = []
        
        # Text length
        features.append(len(text) / 100.0)  # Normalize
        
        # Number of words
        words = self._tokenize(text)
        features.append(len(words) / 20.0)  # Normalize
        
        # Category indicators
        categories = ['chair', 'table', 'car', 'airplane', 'lamp', 'sofa', 'bed']
        for category in categories:
            features.append(1.0 if category in text.lower() else 0.0)
        
        # Material indicators
        materials = ['wooden', 'metal', 'plastic', 'glass', 'leather']
        for material in materials:
            features.append(1.0 if material in text.lower() else 0.0)
        
        # Color indicators
        colors = ['red', 'blue', 'white', 'black', 'green', 'yellow']
        for color in colors:
            features.append(1.0 if color in text.lower() else 0.0)
        
        # Shape indicators
        shapes = ['round', 'square', 'rectangular', 'long', 'short', 'tall']
        for shape in shapes:
            features.append(1.0 if shape in text.lower() else 0.0)
        
        return features

# Test the improved text encoder
def test_improved_encoder():
    """Test the improved text encoder."""
    print("Testing Improved Text Encoder...")
    
    encoder = ImprovedTextEncoder()
    
    # Test different texts
    test_texts = [
        "a wooden chair",
        "a red car",
        "a white airplane",
        "a modern glass table"
    ]
    
    print("Text encodings:")
    for text in test_texts:
        embedding = encoder.encode_text(text)
        print(f"  '{text}': {embedding.shape}, mean: {embedding.mean():.3f}, std: {embedding.std():.3f}")
    
    # Check if different texts produce different embeddings
    embeddings = [encoder.encode_text(text) for text in test_texts]
    
    print("\nEmbedding differences:")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            diff = torch.norm(embeddings[i] - embeddings[j]).item()
            print(f"  '{test_texts[i]}' vs '{test_texts[j]}': {diff:.3f}")

if __name__ == "__main__":
    test_improved_encoder()
