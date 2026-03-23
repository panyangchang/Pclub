"""Pytest configuration — add repo root to sys.path."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
