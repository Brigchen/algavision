#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch Microscope Cut Viewer — Interactive Segmentation Preview"""
import sys, os, subprocess
from pathlib import Path

src = Path(__file__).parent / "src"
env = {**os.environ, "PYTHONPATH": str(src)}
subprocess.run([sys.executable, str(src / "microscope_cut_viewer.py")], env=env)
