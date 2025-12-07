"""
Setup script for GuruAI application.
Run this script to set up the development environment.
"""
import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        'uploads',
        'ai_models',
        'chroma_db',
        'ncert_content',
        'diagrams',
        'previous_papers',
        'logs',
        'user_data'
    ]
    
    print("Creating application directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")
    print()


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")
    print()


def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path('venv')
    
    if venv_path.exists():
        print("Virtual environment already exists.")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping virtual environment creation.")
            return
        print("Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    print("Creating virtual environment...")
    subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    print("✓ Virtual environment created")
    print()


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    
    # Determine the pip executable path
    if sys.platform == 'win32':
        pip_path = Path('venv') / 'Scripts' / 'pip.exe'
    else:
        pip_path = Path('venv') / 'bin' / 'pip'
    
    if not pip_path.exists():
        print("Error: Virtual environment not found. Please create it first.")
        sys.exit(1)
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
    
    # Install requirements
    print("Installing packages from requirements.txt...")
    subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
    print("✓ Requirements installed")
    print()


def create_env_file():
    """Create a .env file if it doesn't exist."""
    env_path = Path('.env')
    
    if env_path.exists():
        print(".env file already exists.")
        return
    
    print("Creating .env file...")
    env_content = """# GuruAI Environment Configuration

# Flask Configuration
SECRET_KEY=change-this-to-a-random-secret-key-in-production
FLASK_ENV=development

# Database
DATABASE_URL=sqlite:///guruai.db

# Tesseract OCR Path (update if needed)
# Windows: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
# Linux/Mac: TESSERACT_CMD=/usr/bin/tesseract
TESSERACT_CMD=tesseract
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✓ .env file created")
    print()


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Activate the virtual environment:")
    
    if sys.platform == 'win32':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Update the .env file with your configuration")
    print("\n3. Run the application:")
    print("   python app.py")
    print("\n4. Open your browser and navigate to:")
    print("   http://localhost:5001")
    print("\n" + "="*60 + "\n")


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("GuruAI Setup Script")
    print("="*60 + "\n")
    
    try:
        check_python_version()
        create_directories()
        create_virtual_environment()
        install_requirements()
        create_env_file()
        print_next_steps()
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
