#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch SEGMAN — Image Segmentation"""
import sys, os, subprocess
from pathlib import Path

src = Path(__file__).parent / "src"
env = {**os.environ, "PYTHONPATH": str(src)}
subprocess.run([sys.executable, str(src / "segman.py")], env=env)
