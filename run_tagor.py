#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch TAGOR — Image Tagging & Labeling"""
import sys, os, subprocess
from pathlib import Path

src = Path(__file__).parent / "src"
env = {**os.environ, "PYTHONPATH": str(src)}
subprocess.run([sys.executable, str(src / "tagor.py")], env=env)
