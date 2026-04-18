"""
Allow running as: python -m showdown.cli [args]
"""
import sys
from .cli import main

sys.exit(main())
