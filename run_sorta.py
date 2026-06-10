#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch SORTA — Plankton Identification & Classification"""
import sys, os, subprocess
from pathlib import Path

src = Path(__file__).parent / "src"
env = {**os.environ, "PYTHONPATH": str(src)}
subprocess.run([sys.executable, str(src / "sorta.py")], env=env)
