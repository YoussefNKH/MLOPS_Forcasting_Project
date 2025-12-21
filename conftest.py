import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add backend directory to path for backend tests
backend_dir = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_dir)
