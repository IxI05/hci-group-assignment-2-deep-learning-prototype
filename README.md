# Farm2Energy HCI Assignment 2: Deep Learning Detection

This repository contains one prototype feature for the Farm2Energy HCI Group Assignment 2 project.

The feature implemented here is **Deep Learning Detection**, focused on detecting manure from a camera feed or uploaded image and estimating useful output values for the Farm2Energy concept.

This is not the full Farm2Energy application. It is the coded prototype feature for the assignment requirement to develop one working feature, upload it to GitHub, and include the GitHub link/code evidence in the report.

## Assignment Context

Farm2Energy is an HCI project concept that supports livestock waste reporting, collection, energy estimation, rewards, and sustainability tracking.

For Assignment 2, the proposed system upgrades include:

- Deep Learning Detection
- Quantum Route Optimization
- AR/XR Energy Visualization

This repository represents the **Deep Learning Detection** upgrade only.

## Features

- Live camera preview
- Local image upload
- Brown pixel segmentation with contour detection
- Linear regression prediction for waste weight
- Biogas and revenue estimates
- Voice/audio feedback after analysis

## Prototype Purpose

The prototype demonstrates how a farmer could use camera-based detection to analyze manure and receive estimated values such as:

- Detected waste density
- Estimated manure weight
- Estimated biogas production
- Estimated financial return

## Requirements

- Python 3
- A working camera for live capture
- Python packages listed in `requirements.txt`

## Run

On macOS, double-click:

```sh
run_app.command
```

Or run from the project folder:

```sh
python3 "HCI-images-deep learning/test_2.py"
```

## Install Dependencies

```sh
python3 -m pip install -r requirements.txt
```
