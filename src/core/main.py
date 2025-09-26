"""
Text2PointCloud - Core Application Logic

This module orchestrates the main application pipeline.
"""

from src.text_processor import TextProcessor
from src.shape_generator import ShapeGenerator
from src.visualization import Visualizer

class Text2PointCloud:
    """
    Main application class that orchestrates the text-to-point-cloud pipeline.
    """
    
    def __init__(self):
        """Initialize the main application."""
        self.text_processor = TextProcessor()
        self.shape_generator = ShapeGenerator()
        self.visualizer = Visualizer()
    
    def process_input(self, text_input: str):
        """
        Process text input and generate point cloud visualization.
        
        Args:
            text_input: The input text to process
        """
        # TODO: Implement main pipeline
        # 1. Analyze text
        # 2. Generate shape
        # 3. Visualize result
        pass
    
    def run(self):
        """Run the interactive application."""
        # TODO: Implement interactive mode
        pass

if __name__ == "__main__":
    app = Text2PointCloud()
    app.run()
