#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch MODLER v2 — Model Training & Active Learning"""
import sys, os, subprocess
from pathlib import Path

src = Path(__file__).parent / "src"
env = {**os.environ, "PYTHONPATH": str(src)}
subprocess.run([sys.executable, str(src / "modler_v2.py")], env=env)
