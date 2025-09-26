#!/usr/bin/env python3
"""
Text2PointCloud - Quick Start Script

This script provides easy access to all interfaces.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("🎯" + "="*60 + "🎯")
    print("           Text2PointCloud - Quick Start")
    print("    Generate 3D Point Clouds from Text Descriptions")
    print("🎯" + "="*60 + "🎯")
    print()

def check_model():
    """Check if the trained model exists."""
    model_path = Path("data/comprehensive_trained_model.pth")
    if model_path.exists():
        print("✅ Trained model found!")
        return True
    else:
        print("⚠️  Trained model not found. Please run training first:")
        print("   python3 test_comprehensive_shapenet_training.py")
        return False

def show_menu():
    """Show the main menu."""
    print("Choose an interface:")
    print()
    print("1. 🌐 Web Interface (Flask + Plotly)")
    print("   - Beautiful web interface")
    print("   - Interactive 3D visualization")
    print("   - Real-time generation")
    print("   - ✨ NEW: Actual object outlines!")
    print()
    print("2. 🖥️  Desktop Interface (Tkinter + Matplotlib)")
    print("   - Native desktop application")
    print("   - 3D visualization with matplotlib")
    print("   - Offline operation")
    print()
    print("3. 💻 Command Line Interface")
    print("   - Simple command line tool")
    print("   - Batch processing")
    print("   - Script integration")
    print()
    print("4. 🧪 Test Generation (Quick Test)")
    print("   - Generate a few test point clouds")
    print("   - Verify everything is working")
    print()
    print("0. Exit")
    print()

def run_web_interface():
    """Run the web interface."""
    print("🌐 Starting Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, "web_interface_outline.py"])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def run_desktop_interface():
    """Run the desktop interface."""
    print("🖥️  Starting Desktop Interface...")
    print("🖼️  Desktop application window will open")
    print("🛑 Close the window to exit")
    print()
    
    try:
        subprocess.run([sys.executable, "desktop_interface.py"])
    except Exception as e:
        print(f"❌ Error starting desktop interface: {e}")

def run_cli_interface():
    """Run the command line interface."""
    print("💻 Command Line Interface")
    print("="*40)
    print()
    print("Usage examples:")
    print("  python3 cli_interface.py 'a wooden chair'")
    print("  python3 cli_interface.py 'a red sports car' --save-plot output.png")
    print("  python3 cli_interface.py --interactive")
    print()
    
    # Get text input
    text = input("Enter text description: ").strip()
    if not text:
        print("No text entered. Exiting.")
        return
    
    # Run CLI
    try:
        subprocess.run([sys.executable, "cli_interface.py", text])
    except Exception as e:
        print(f"❌ Error running CLI: {e}")

def run_test_generation():
    """Run a quick test generation."""
    print("🧪 Running Quick Test Generation...")
    print("="*40)
    
    test_texts = [
        "a wooden chair with four legs",
        "a red sports car with two doors",
        "a white commercial airplane",
        "a modern glass table"
    ]
    
    try:
        for text in test_texts:
            print(f"\n🔄 Generating: '{text}'")
            result = subprocess.run([
                sys.executable, "cli_interface.py", text, "--no-plot"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Success!")
                # Extract quality score from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Compactness:' in line:
                        print(f"   Quality: {line.strip()}")
                        break
            else:
                print(f"❌ Error: {result.stderr}")
                
    except Exception as e:
        print(f"❌ Error running test: {e}")

def main():
    """Main function."""
    print_banner()
    
    # Check if model exists
    if not check_model():
        print("\nPlease train the model first, then run this script again.")
        return
    
    print()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_web_interface()
            elif choice == '2':
                run_desktop_interface()
            elif choice == '3':
                run_cli_interface()
            elif choice == '4':
                run_test_generation()
            else:
                print("❌ Invalid choice. Please enter 0-4.")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
