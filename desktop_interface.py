"""
Text2PointCloud - Desktop Interface

A Tkinter desktop application for text-to-point-cloud generation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os
from pathlib import Path
import threading
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.improved_encoder import ImprovedTextEncoder
from validation.validator import PointCloudValidator
import torch

class Text2PointCloudDesktopApp:
    """
    Desktop application for text-to-point-cloud generation.
    """
    
    def __init__(self, root):
        """Initialize the desktop app."""
        self.root = root
        self.root.title("Text2PointCloud - Generate 3D Point Clouds from Text")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.text_encoder = ImprovedTextEncoder()
        self.validator = PointCloudValidator()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Current point cloud data
        self.current_points = None
        self.current_text = ""
        
        # Load the trained model
        self.load_model()
        
        # Create UI
        self.create_ui()
    
    def load_model(self):
        """Load the trained model."""
        try:
            from test_comprehensive_shapenet_training import TextToPointCloudModel
            
            # Create model
            self.model = TextToPointCloudModel(self.text_encoder)
            
            # Load trained weights
            model_path = "data/comprehensive_trained_model.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print("✓ Trained model loaded successfully")
            else:
                print("⚠️ Trained model not found, using untrained model")
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 Text2PointCloud", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Input and controls
        left_panel = ttk.LabelFrame(main_frame, text="Input & Controls", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Text input
        ttk.Label(left_panel, text="Describe the object:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.text_input = scrolledtext.ScrolledText(left_panel, height=4, width=40, wrap=tk.WORD)
        self.text_input.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.text_input.insert('1.0', "a comfortable wooden dining chair")
        
        # Number of points
        ttk.Label(left_panel, text="Number of points:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.num_points_var = tk.StringVar(value="1024")
        num_points_entry = ttk.Entry(left_panel, textvariable=self.num_points_var, width=10)
        num_points_entry.grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Point Cloud", 
                                      command=self.generate_point_cloud)
        self.generate_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", 
                                   command=self.clear_results)
        self.clear_btn.grid(row=0, column=1)
        
        # Examples
        examples_frame = ttk.LabelFrame(left_panel, text="Examples", padding="5")
        examples_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        examples = [
            "a wooden chair with four legs",
            "a red sports car with two doors", 
            "a white commercial airplane",
            "a modern glass table",
            "a comfortable leather sofa",
            "a modern desk lamp"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(examples_frame, text=example, 
                           command=lambda e=example: self.set_text(e))
            btn.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=2, pady=2)
        
        # Configure example buttons to expand
        for i in range(2):
            examples_frame.columnconfigure(i, weight=1)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(left_panel, textvariable=self.status_var, 
                                foreground='blue')
        status_label.grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        
        # Right panel - Visualization and results
        right_panel = ttk.LabelFrame(main_frame, text="3D Visualization & Results", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial empty plot
        self.plot_empty()
        
        # Results frame
        results_frame = ttk.LabelFrame(right_panel, text="Generation Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Results text
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=60)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        # Initial empty plot
        self.plot_empty()
    
    def set_text(self, text):
        """Set text in the input field."""
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', text)
    
    def generate_point_cloud(self):
        """Generate point cloud from text input."""
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text")
            return
        
        try:
            num_points = int(self.num_points_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of points")
            return
        
        # Disable button and show status
        self.generate_btn.config(state='disabled')
        self.status_var.set("Generating...")
        self.root.update()
        
        # Generate in separate thread to avoid blocking UI
        thread = threading.Thread(target=self._generate_worker, args=(text, num_points))
        thread.daemon = True
        thread.start()
    
    def _generate_worker(self, text, num_points):
        """Worker function for point cloud generation."""
        try:
            if self.model is None:
                self.root.after(0, lambda: self._show_error("Model not loaded"))
                return
            
            # Generate point cloud
            self.model.eval()
            with torch.no_grad():
                point_cloud = self.model([text])
                point_cloud = point_cloud.squeeze(0).cpu().numpy()
            
            # Convert to list of tuples
            points = [(float(x), float(y), float(z)) for x, y, z in point_cloud]
            
            # Validate
            validation_results = self.validator.validate_generation(
                np.array(points), text
            )
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_results(text, points, validation_results))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Error: {str(e)}"))
    
    def _update_results(self, text, points, validation_results):
        """Update the UI with generation results."""
        self.current_text = text
        self.current_points = points
        
        # Update plot
        self.plot_point_cloud(points, text)
        
        # Update results text
        self.update_results_text(text, points, validation_results)
        
        # Re-enable button and update status
        self.generate_btn.config(state='normal')
        self.status_var.set("Ready - Point cloud generated!")
    
    def _show_error(self, error_msg):
        """Show error message."""
        self.generate_btn.config(state='normal')
        self.status_var.set("Error")
        messagebox.showerror("Error", error_msg)
    
    def plot_point_cloud(self, points, text):
        """Plot the point cloud in 3D."""
        self.ax.clear()
        
        if not points:
            self.plot_empty()
            return
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Create 3D scatter plot
        scatter = self.ax.scatter(x_coords, y_coords, z_coords, 
                                 c=z_coords, cmap='viridis', s=20, alpha=0.7)
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"Generated Point Cloud: '{text}'")
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        # Add colorbar
        self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5, label='Z Coordinate')
        
        # Refresh canvas
        self.canvas.draw()
    
    def plot_empty(self):
        """Plot empty 3D space."""
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("3D Point Cloud Visualization")
        self.ax.text(0, 0, 0, "Enter text and click 'Generate Point Cloud'", 
                    ha='center', va='center', fontsize=12, color='gray')
        self.canvas.draw()
    
    def update_results_text(self, text, points, validation_results):
        """Update the results text area."""
        self.results_text.delete('1.0', tk.END)
        
        # Basic info
        results = f"Generated Point Cloud for: '{text}'\n"
        results += f"Number of points: {len(points)}\n"
        results += f"Model device: {self.device}\n"
        results += "=" * 50 + "\n\n"
        
        # Validation results
        if validation_results:
            results += "VALIDATION RESULTS:\n"
            results += "-" * 30 + "\n"
            
            # Category accuracy
            if 'category_accuracy' in validation_results:
                cat_acc = validation_results['category_accuracy']
                results += f"Predicted Category: {cat_acc.get('predicted_category', 'Unknown')}\n"
                results += f"Confidence: {cat_acc.get('confidence', 0):.3f}\n\n"
            
            # Quality metrics
            if 'quality_metrics' in validation_results:
                quality = validation_results['quality_metrics']
                results += "QUALITY METRICS:\n"
                results += f"  Compactness: {quality.get('compactness', 0):.3f}\n"
                results += f"  Spread: {quality.get('spread', 0):.3f}\n"
                results += f"  Density: {quality.get('density', 0):.3f}\n\n"
            
            # Shape metrics
            if 'shape_metrics' in validation_results:
                shape = validation_results['shape_metrics']
                results += "SHAPE METRICS:\n"
                dims = shape.get('dimensions', [0, 0, 0])
                results += f"  Dimensions: {dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f}\n"
                results += f"  Volume: {shape.get('volume', 0):.3f}\n"
                results += f"  Surface Area: {shape.get('surface_area', 0):.3f}\n\n"
            
            # Text consistency
            if 'text_consistency' in validation_results:
                text_cons = validation_results['text_consistency']
                results += "TEXT CONSISTENCY:\n"
                results += f"  Overall: {text_cons.get('overall_consistency', 0):.3f}\n"
                results += f"  Semantic: {text_cons.get('semantic_consistency', 0):.3f}\n"
                results += f"  Structural: {text_cons.get('structural_consistency', 0):.3f}\n\n"
        
        # Point cloud statistics
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            
            results += "POINT CLOUD STATISTICS:\n"
            results += "-" * 30 + "\n"
            results += f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}\n"
            results += f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}\n"
            results += f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}\n"
            results += f"Center: ({np.mean(x_coords):.3f}, {np.mean(y_coords):.3f}, {np.mean(z_coords):.3f})\n"
        
        self.results_text.insert('1.0', results)
    
    def clear_results(self):
        """Clear all results."""
        self.text_input.delete('1.0', tk.END)
        self.num_points_var.set("1024")
        self.results_text.delete('1.0', tk.END)
        self.plot_empty()
        self.status_var.set("Ready")
        self.current_points = None
        self.current_text = ""

def main():
    """Main function to run the desktop app."""
    root = tk.Tk()
    app = Text2PointCloudDesktopApp(root)
    
    print("🚀 Starting Text2PointCloud Desktop Interface...")
    print("🖥️  Desktop application is now running!")
    print("🛑 Close the window to exit")
    
    root.mainloop()

if __name__ == "__main__":
    main()
