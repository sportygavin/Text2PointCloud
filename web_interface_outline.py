"""
Text2PointCloud - Updated Web Interface with Outline Generation

A Flask web application for text-to-point-cloud generation with actual object outlines.
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.improved_encoder import ImprovedTextEncoder
from validation.validator import PointCloudValidator
from outline_model import OutlineBasedModel
import torch

app = Flask(__name__)

class Text2PointCloudWebApp:
    """
    Web application for text-to-point-cloud generation with outline generation.
    """
    
    def __init__(self):
        """Initialize the web app."""
        self.text_encoder = ImprovedTextEncoder()
        self.validator = PointCloudValidator()
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the outline-based model
        self.load_model()
    
    def load_model(self):
        """Load the outline-based model."""
        try:
            # Create outline-based model
            self.model = OutlineBasedModel(self.text_encoder)
            self.model.to(self.device)
            self.model.eval()
            print("✓ Outline-based model loaded successfully")
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def generate_point_cloud(self, text: str, num_points: int = 1024):
        """
        Generate point cloud from text.
        
        Args:
            text: Input text description
            num_points: Number of points in generated cloud
            
        Returns:
            Dictionary with point cloud data and validation results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'points': [],
                'validation': {}
            }
        
        try:
            # Generate point cloud using outline-based model
            points = self.model.generate_point_cloud(text)
            
            # Limit to requested number of points
            if len(points) > num_points:
                points = points[:num_points]
            
            # Validate
            validation_results = self.validator.validate_generation(
                np.array(points), text
            )
            
            return {
                'points': points,
                'validation': validation_results,
                'text': text,
                'num_points': len(points)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'points': [],
                'validation': {}
            }
    
    def create_plotly_figure(self, points, text, validation_results):
        """
        Create Plotly figure for 3D visualization.
        
        Args:
            points: List of (x, y, z) coordinate tuples
            text: Input text description
            validation_results: Validation results dictionary
            
        Returns:
            Plotly figure object
        """
        if not points:
            return go.Figure()
        
        # Extract coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=z_coords,  # Color by z-coordinate
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Z Coordinate")
            ),
            text=[f"Point {i}" for i in range(len(points))],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title=f"Generated Object Outline: '{text}'",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig

# Create app instance
web_app = Text2PointCloudWebApp()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate point cloud from text input."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        num_points = int(data.get('num_points', 1024))
        
        if not text:
            return jsonify({'error': 'Please enter some text'})
        
        # Generate point cloud
        result = web_app.generate_point_cloud(text, num_points)
        
        if 'error' in result:
            return jsonify(result)
        
        # Create Plotly figure
        fig = web_app.create_plotly_figure(
            result['points'], 
            result['text'], 
            result['validation']
        )
        
        # Convert to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Prepare response
        response = {
            'graphJSON': graphJSON,
            'validation': result['validation'],
            'text': result['text'],
            'num_points': result['num_points']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': web_app.model is not None,
        'device': web_app.device
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text2PointCloud - Generate 3D Object Outlines from Text</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .input-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        .loading.show {
            display: block;
        }
        .results {
            margin-top: 30px;
        }
        .validation-info {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4caf50;
        }
        .validation-info h3 {
            margin: 0 0 15px 0;
            color: #2e7d32;
        }
        .validation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .validation-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .validation-item h4 {
            margin: 0 0 8px 0;
            color: #333;
            font-size: 14px;
        }
        .validation-item p {
            margin: 0;
            color: #666;
            font-size: 18px;
            font-weight: 600;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
            margin: 20px 0;
        }
        .examples {
            background: #f0f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .examples h3 {
            margin: 0 0 15px 0;
            color: #333;
        }
        .example-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .example-tag {
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            transition: background 0.3s;
            font-size: 14px;
        }
        .example-tag:hover {
            background: #5a6fd8;
        }
        .info-box {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }
        .info-box h4 {
            margin: 0 0 10px 0;
            color: #1976d2;
        }
        .info-box p {
            margin: 0;
            color: #424242;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Text2PointCloud</h1>
            <p>Generate 3D Object Outlines from Text Descriptions</p>
        </div>
        
        <div class="content">
            <div class="info-box">
                <h4>✨ New: Actual Object Outlines!</h4>
                <p>This version generates actual object outlines instead of random blobs. Try describing chairs, cars, tables, and more!</p>
            </div>
            
            <div class="input-section">
                <div class="input-group">
                    <label for="textInput">Describe the object you want to generate:</label>
                    <input type="text" id="textInput" placeholder="e.g., a wooden chair with four legs" value="a wooden chair with four legs">
                </div>
                
                <div class="input-group">
                    <label for="numPoints">Number of points (100-2048):</label>
                    <input type="number" id="numPoints" value="1024" min="100" max="2048">
                </div>
                
                <div class="button-group">
                    <button onclick="generatePointCloud()">Generate Object Outline</button>
                    <button onclick="clearResults()">Clear</button>
                </div>
                
                <div class="loading" id="loading">
                    <p>🔄 Generating object outline... Please wait...</p>
                </div>
            </div>
            
            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <div class="example-tags">
                    <span class="example-tag" onclick="setText('a wooden chair with four legs')">Wooden Chair</span>
                    <span class="example-tag" onclick="setText('a red sports car with two doors')">Red Sports Car</span>
                    <span class="example-tag" onclick="setText('a white commercial airplane')">White Airplane</span>
                    <span class="example-tag" onclick="setText('a modern glass table')">Glass Table</span>
                    <span class="example-tag" onclick="setText('a comfortable leather sofa')">Leather Sofa</span>
                    <span class="example-tag" onclick="setText('a modern desk lamp')">Desk Lamp</span>
                    <span class="example-tag" onclick="setText('a wooden bed with headboard')">Wooden Bed</span>
                    <span class="example-tag" onclick="setText('a bicycle with two wheels')">Bicycle</span>
                </div>
            </div>
            
            <div class="results" id="results" style="display: none;">
                <div class="validation-info" id="validationInfo">
                    <h3>📊 Generation Results</h3>
                    <div class="validation-grid" id="validationGrid">
                        <!-- Validation results will be inserted here -->
                    </div>
                </div>
                
                <div id="plotlyDiv">
                    <!-- Plotly visualization will be inserted here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        function setText(text) {
            document.getElementById('textInput').value = text;
        }
        
        function generatePointCloud() {
            const text = document.getElementById('textInput').value.trim();
            const numPoints = parseInt(document.getElementById('numPoints').value);
            
            if (!text) {
                alert('Please enter some text');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').style.display = 'none';
            
            // Make request
            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    num_points: numPoints
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Show results
                showResults(data);
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                showError('Error: ' + error.message);
            });
        }
        
        function showResults(data) {
            // Show validation info
            const validationGrid = document.getElementById('validationGrid');
            const validation = data.validation;
            
            validationGrid.innerHTML = `
                <div class="validation-item">
                    <h4>Predicted Category</h4>
                    <p>${validation.category_accuracy?.predicted_category || 'Unknown'}</p>
                </div>
                <div class="validation-item">
                    <h4>Quality Score</h4>
                    <p>${(validation.quality_metrics?.compactness || 0).toFixed(3)}</p>
                </div>
                <div class="validation-item">
                    <h4>Text Consistency</h4>
                    <p>${(validation.text_consistency?.overall_consistency || 0).toFixed(3)}</p>
                </div>
                <div class="validation-item">
                    <h4>Points Generated</h4>
                    <p>${data.num_points}</p>
                </div>
                <div class="validation-item">
                    <h4>Dimensions</h4>
                    <p>${validation.shape_metrics?.dimensions?.map(d => d.toFixed(2)).join(' × ') || 'N/A'}</p>
                </div>
                <div class="validation-item">
                    <h4>Spread</h4>
                    <p>${(validation.shape_metrics?.spread || 0).toFixed(3)}</p>
                </div>
            `;
            
            // Show plotly visualization
            const graphJSON = JSON.parse(data.graphJSON);
            Plotly.newPlot('plotlyDiv', graphJSON.data, graphJSON.layout, {responsive: true});
            
            // Show results section
            document.getElementById('results').style.display = 'block';
        }
        
        function showError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="error">${message}</div>`;
            resultsDiv.style.display = 'block';
        }
        
        function clearResults() {
            document.getElementById('results').style.display = 'none';
            document.getElementById('textInput').value = '';
            document.getElementById('numPoints').value = '1024';
        }
        
        // Allow Enter key to generate
        document.getElementById('textInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                generatePointCloud();
            }
        });
    </script>
</body>
</html>
    '''
    
    # Write template file
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("🚀 Starting Text2PointCloud Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
