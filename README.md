# Farm2Energy HCI Assignment 2: Deep Learning Detection

This repository contains the coded prototype feature for the Farm2Energy HCI Group Assignment 2 project.

The implemented feature is **Deep Learning Detection**. It is not the full Farm2Energy application. It demonstrates the report waste flow where a farmer can use AI image detection to estimate manure quantity and apply the detected information back into the Report Waste interface.

## Feature Scope

- Report Waste interface for waste type, estimated quantity, farm location, and condition notes
- AI detection screen opened from the report form
- Camera capture or local JPG/PNG upload for image analysis
- Manure segmentation using OpenCV color detection
- Simple machine learning estimate for manure mass
- Detection result fields:
  - manure type
  - manure condition
  - confidence rate
  - mass estimate
  - biogas estimate
  - CO2e reduction estimate
  - estimated return
- Apply Information button to fill the Report Waste form automatically
- Retake Image flow that clears the analyzed image, pauses audio, and reopens the camera
- Scrollable interface for smaller windows
- macOS and Windows launcher files

## Requirements

- Python 3.10 or later
- Python packages listed in `requirements.txt`
- Camera access for live capture

If a camera is not available, the prototype can still analyze uploaded JPG/PNG images.

## Install Dependencies

macOS:

```sh
python3 -m pip install -r requirements.txt
```

Windows:

```bat
py -3 -m pip install -r requirements.txt
```

## Run on macOS

Double-click:

```sh
run_app.command
```

Or run manually:

```sh
python3 "HCI-images-deep learning/test_2.py"
```

## Run on Windows

Double-click:

```bat
run_app.bat
```

Or run manually:

```bat
py -3 "HCI-images-deep learning\test_2.py"
```

## Notes

- The Submit Report button only shows a prototype submitted status. It does not send data to a real system.
- The feature starts on the Report Waste form. Click **Use AI Detection** to open the image detection interface.
- After analysis, click **Apply Information to Report Waste** to copy the AI result back into the form.
