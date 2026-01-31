"""Performance tests package initialization"""

# Import fixtures from main fixtures module
import sys
import os

# Add parent directory to path for fixture imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures import *
