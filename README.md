# StyleGAN Face Generator

A desktop application for generating photorealistic human faces using **StyleGAN v1** (FFHQ 1024x1024), with controls for **age range** and **gender** filters. Built with PyQt5 and TensorFlow 1.15, runs fully on **CPU** (no GPU required).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Optional: Age Calibration (Recommended)](#optional-age-calibration-recommended)
- [Project Structure](#project-structure)
- [Filters Reference](#filters-reference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.7.x** | Strictly 3.7 — TF 1.15 does not support 3.8+ |
| **pip** | any recent | bundled with Python |
| **Internet connection** | — | First run downloads the model (~350 MB) and boundary vectors |
| **RAM** | >= 8 GB | Model loading + inference is memory-heavy on CPU |
| **Disk** | >= 1.5 GB free | Model + cache files |

> **Windows note:** A Python 3.7.9 installer (`python-3.7.9-amd64.exe`) is included in the repo for convenience.

---

## Setup

### 1. Clone or unzip the repository

```powershell
git clone <repo-url> FaceGenerator
cd FaceGenerator
```

### 2. Create a Python 3.7 virtual environment

```powershell
py -3.7 -m venv .venv37

# Activate it (Windows PowerShell)
.\.venv37\Scripts\Activate.ps1
```

> If you get a scripts execution error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install optional dependencies (for age validation)

These enable InsightFace closed-loop age validation and the age calibration script.
The app works without them, but age accuracy is lower.

```powershell
pip install insightface onnxruntime scikit-learn
```

---

## Running the App

```powershell
# Make sure the venv is activated first
.\.venv37\Scripts\Activate.ps1

python app.py
```

On first launch the app will automatically:
1. Download the pretrained StyleGAN FFHQ model (~350 MB) into `cache/`
2. Download the age and gender boundary vectors from InterFaceGAN into `cache/`

This only happens once. Subsequent launches are instant.

### Using the app

1. **Output Directory** - click **Browse** to choose where generated images are saved (defaults to `output/`)
2. **Number of Images** - how many faces to generate (1-1000)
3. **Age Range** - select a bucket or leave as **Any**
4. **Gender** - Male, Female, or Any
5. Click **Generate Images**

Generated images are saved as `generated_0000.png`, `generated_0001.png`, ... in the output directory.
A live preview is shown in the right panel.

---

## Optional: Age Calibration (Recommended)

By default the app uses a pre-trained SVM boundary vector as a fallback for age control.
For significantly better age accuracy, run the one-time calibration script:

```powershell
# Requires insightface, onnxruntime, scikit-learn (see step 4 above)
python scripts\fit_age_direction.py --samples 2000
```

This takes **~20-30 minutes on CPU**. When it finishes it writes `cache/age_regression.npz`.
The next time you start the app you will see:

```
[LatentDirections] Using calibrated ridge-regression age direction.
```

### Calibration options

| Flag | Default | Description |
|---|---|---|
| `--samples` | `2000` | Number of faces to sample. More = more accurate, but slower |
| `--seed` | `0` | RNG seed for reproducibility |
| `--alpha` | `0.001` | Ridge regularisation strength |
| `--out` | `cache/age_regression.npz` | Output file path |

---

## Project Structure

```
FaceGenerator/
|-- app.py                    # PyQt5 UI -- main entry point
|-- engine.py                 # StyleGAN loading, LatentDirections, generation threads
|-- config.py                 # Paths (cache dir, model dir)
|-- style.qss                 # Qt stylesheet for the UI
|-- requirements.txt          # Pinned Python dependencies
├── ui/
│   ├── window.py                   # FaceGeneratorApp: __init__ and wiring only
│   │                               
│   ├── sidebar.py                  # build_sidebar(window): output, filters, status card, Generate button
│   │
│   ├── preview_panel.py             # Main preview area: toolbar, canvas, thumbnail strip
│   │
│   ├── widgets.py                  # Reusable custom Qt widgets 
│   │
│   ├── icons.py                    # SVG icon system
│   │
│   ├── qt_helpers.py               # Qt conversion/helper functions
│   │
│   ├── theme.py                    # stylesheet loading and the dropdown-arrow icon files
│   │
│   ├── paths.py                    # project folder and default output folder
│   │
│   ├── constants.py                # UI constants
│   │
│   ├── controls.py                 # Control behaviour: image-count stepper, gender, reset
│   │
│   ├── status.py                   # Status card behaviour
│   │
│   ├── generation.py               # Generation workflow: model loading, output folder, generate workflow
│   │
│   ├── gallery.py                  # Gallery behaviour: preview navigation and thumbnail strip
│   │
│   └── image_actions.py            # Image operations: Save as, Copy, Delete
|
|-- scripts/
|   `-- fit_age_direction.py  # One-time age calibration script
|
|-- dnnlib/                   # NVIDIA dnnlib (TF graph utilities)
|-- training/                 # StyleGAN training code (not needed to run the app)
|-- metrics/                  # FID / precision-recall metrics
|
|-- cache/                    # Auto-created on first run
|   |-- karras2019stylegan-ffhq-1024x1024.pkl  # Pretrained model (auto-downloaded)
|   |-- stylegan_ffhq_age_boundary.npy          # SVM boundary (auto-downloaded)
|   |-- stylegan_ffhq_gender_boundary.npy       # SVM boundary (auto-downloaded)
|   `-- age_regression.npz                      # Created by fit_age_direction.py
|
`-- output/                   # Default output folder for generated images
```

---

## Filters Reference

### Age Ranges

| Label | Description |
|---|---|
| Any | No age edit applied |
| 1-11 years old | Child |
| 12-18 years old | Teenager |
| 19-25 years old | Young adult |
| 26-35 years old | Adult |
| 35-50 years old | Middle-aged |
| 50+ years old | Senior |

> Age filtering works in two modes:
> - **Regression mode** (after running `fit_age_direction.py`): shifts the latent by calibrated real-year units.
> - **SVM fallback** (default): uses a hand-tuned boundary offset. Results are good but less precise.

### Gender

Steers the face toward male or female using the InterFaceGAN gender boundary.
Selecting **Any** leaves gender untouched (random from the model).

---

## Troubleshooting

### App will not start / import errors
- Make sure you activated `.venv37` before running `python app.py`.
- Confirm Python version is 3.7: `python --version`

### "Model loading..." never finishes
- Check your internet connection -- the model (~350 MB) must download on first run.
- If the download was interrupted, delete the partial file in `cache/` and restart.

### Images look psychedelic / corrupted
- This was caused by the age latent edit going out of distribution.
  It is fixed in the current version via delta clamping in `engine.py`.

### `fit_age_direction.py` crashes with `UnimplementedError: Depthwise convolution`
- Fixed in the current version -- the depthwise NCHW->NHWC patch is now included in the script.

### InsightFace `ValueError: operands could not be broadcast ... shapes (18,) (32,)`
- Was a `det_size=112` issue (not a multiple of 32). Fixed -- current version uses `det_size=160`.

### TensorFlow CUDA / GPU warnings
```
Could not load dynamic library 'cudart64_100.dll'
```
This is harmless on a CPU-only machine. TF 1.15 always attempts to find CUDA.
The app runs correctly on CPU regardless.

### PowerShell execution policy error when activating venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```


