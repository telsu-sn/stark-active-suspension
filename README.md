# Stark Active Suspension Controller

This repository contains the final solution for the Stark Active Suspension
Kaggle competition.

## Overview
We design a semi-active suspension controller for a 2-DOF quarter-car model.
The controller uses frequency-selective skyhook damping with acceleration
feedback to minimize body displacement and jerk.

Final leaderboard score: **57.6**

## Requirements
- Python 3.8+
- numpy
- pandas

Install dependencies:
pip install -r requirements.txt

## How to Reproduce Results

1. Download `road_profiles.csv` from the Kaggle competition page
2. Place it in the same directory as `run_simulation.py`
3. Run:
   python run_simulation.py
4. This generates `submission.csv` ready for Kaggle upload

## Files
- `run_simulation.py`: quarter-car simulation + controller
- `requirements.txt`: dependencies
- `Untitled document(1).pdf`: technical explanation and plots

