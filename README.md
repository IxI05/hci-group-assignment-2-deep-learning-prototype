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
- Confidence rate, CO2e reduction, manure type, and manure condition output
- Report Waste interface for entering estimated quantity, location, and notes
- Apply AI detection information into the Report Waste form
- Voice/audio feedback after analysis

## Prototype Purpose

The prototype demonstrates how a farmer could use camera-based detection to analyze manure and receive estimated values such as:

- Detected waste density
- Estimated manure weight
- Estimated biogas production
- Estimated CO2e reduction
- Estimated financial return

After analysis is completed, the **Use AI Detection Information** button can automatically fill the Report Waste interface with the detected manure type, quantity, and condition notes.

## Requirements

- Python 3.10 or later
- A working camera for live capture
- Python packages listed in `requirements.txt`

Tkinter is included with most Python installs. If the camera is unavailable, the prototype can still analyze uploaded JPG/PNG images.

## Install Dependencies

macOS:

```sh
python3 -m pip install -r requirements.txt
```

Windows:

```bat
py -3 -m pip install -r requirements.txt
```

## Run

macOS, double-click:

```sh
run_app.command
```

Windows, double-click:

```bat
run_app.bat
```

Or run manually from the project folder.

macOS:

```sh
python3 "HCI-images-deep learning/test_2.py"
```

Windows:

```bat
py -3 "HCI-images-deep learning\test_2.py"
```
