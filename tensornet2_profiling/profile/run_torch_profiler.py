#!/usr/bin/env python3
"""Thin convenience wrapper: PyTorch Profiler pass over the harness.
Equivalent to `python harness/run_infer.py --profile <args>`.
Example: conda run -n tn2prof python profile/run_torch_profiler.py --workload B --n 1000
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
sys.path.insert(0, HARNESS)
sys.argv = [os.path.join(HARNESS, "run_infer.py"), "--profile", *sys.argv[1:]]
runpy.run_path(os.path.join(HARNESS, "run_infer.py"), run_name="__main__")
