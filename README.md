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
- [Running the Tests](#running-the-tests)
- [Building the Desktop App and Installer](#building-the-desktop-app-and-installer)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.7.x** | Strictly 3.7 — TF 1.15 does not support 3.8+ |
| **pip** | any recent | bundled with Python |
| **Internet connection** | — | First run downloads the model (~325 MB) and boundary vectors |
| **RAM** | >= 8 GB | Model loading + inference is memory-heavy on CPU |
| **Disk** | >= 1.5 GB free | Model + cache files |

> **Windows note:** Python 3.7.9 for Windows (`python-3.7.9-amd64.exe`, "Windows x86-64 executable installer")
> is available from [python.org](https://www.python.org/downloads/release/python-379/).

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
1. Download the pretrained StyleGAN FFHQ model (~325 MB) into `cache/`
2. Download the age and gender boundary vectors from InterFaceGAN into `cache/`

This only happens once. Subsequent launches are instant.

### Using the app

1. **Output Directory** - click **Browse** to choose where generated images are saved (defaults to `output/`)
2. **Number of Images** - how many faces to generate (1-1000, default 20)
3. **Age Range** - select a bucket or leave as **Any** (default)
4. **Gender** - Male, Female, or Any (default)
5. Click **Generate Images**

**Reset** (next to FILTERS) returns age, gender and number of images to these defaults,
the same state the app starts in.

Generated images are saved as `generated_0000.png`, `generated_0001.png`, ... in the output directory.
Each run continues numbering after the highest existing `generated_NNNN.png`, so earlier images are never overwritten.
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
The next time you generate images, the console and `logs\facegen.log` show:

```
[latents] Using calibrated ridge-regression age direction.
```

### Calibration options

| Flag | Default | Description |
|---|---|---|
| `--samples` | `2000` | Number of faces to sample. More = more accurate, but slower |
| `--seed` | `0` | RNG seed for reproducibility |
| `--alpha` | `0.001` | Ridge regularisation strength |
| `--out` | `cache/age_regression.npz` | Output file path |

### After recalibrating: re-measure the age response

Rendered faces move less than the calibration predicts, so `latents.py` holds a measured
correction (`AGE_RESPONSE`). After running `fit_age_direction.py` again, re-measure it:

```powershell
python scripts\measure_age_response.py
```

It renders real faces (~12 minutes on CPU), prints a new `AGE_RESPONSE` table to paste into
`latents.py`, and reports how many faces land in each age bucket. Use `--fit-samples` and
`--report-samples` to trade accuracy for time.

---

## Project Structure

```
FaceGenerator/
|-- app.py                    # PyQt5 UI -- main entry point
|-- engine.py                 # Qt threads: model loading and generation, reported via signals
|-- latents.py                # Age buckets (AGE_BUCKETS) and the latent age/gender edits
|-- generator.py              # Generation core: age-check retries, batch run, output numbering
|-- model.py                  # Load the pretrained StyleGAN model (download + checksum)
|-- tf_compat.py              # CPU patches so the NCHW model runs on TensorFlow CPU
|-- config.py                 # Paths (cache dir, log dir; different for the installed app)
|-- downloads.py              # Downloaded files, their checksums, verified download helpers
|-- log_setup.py              # Logging: console, logs/facegen.log, quieter TensorFlow, error hook
|-- style.qss                 # Qt stylesheet for the UI
|-- requirements.txt          # Pinned Python dependencies
|-- requirements-dev.txt      # Test dependencies (pytest)
|-- pytest.ini                # Test settings (markers, default selection)
|-- tests/                    # pytest suite (see "Running the Tests")
|-- requirements-build.txt    # Build dependencies (PyInstaller)
|-- packaging/                # Desktop app + installer (see "Building the Desktop App")
|   |-- build.ps1             # Runs the whole build
|   |-- facegen.spec          # PyInstaller bundle definition
|   |-- installer.iss         # Inno Setup installer script
|   |-- logo.png              # App logo (window, taskbar, sidebar header)
|   `-- make_icon.py          # Makes facegen.ico (.exe/installer icon) from logo.png
├── components/
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
|   |-- fit_age_direction.py     # One-time age calibration script
|   `-- measure_age_response.py  # Re-measure AGE_RESPONSE after recalibrating
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
|-- logs/                     # facegen.log (auto-created; see "Where to find logs")
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

## Running the Tests

```powershell
pip install -r requirements-dev.txt   # once: installs pytest
python -m pytest                      # ~100 tests, ~15 s; needs neither the model nor internet
python -m pytest -m model             # tests with the real model in cache/ (~20 s)
python -m pytest -m network           # checks the pinned boundary downloads (needs internet)
```

The default run uses small synthetic boundary/calibration files and a fake generator
network, so it works on a fresh checkout. The UI tests run offscreen; no window appears.

---

## Building the Desktop App and Installer

`packaging\build.ps1` turns the app into a normal Windows program: a
`FaceGenerator.exe` that runs without Python, and a setup wizard that installs it
with a Start-menu shortcut and an uninstaller. Users don't need Python or a terminal.

**One-time setup** (in addition to [Setup](#setup)):

```powershell
.\.venv37\Scripts\python.exe -m pip install -r requirements-build.txt   # PyInstaller
winget install JRSoftware.InnoSetup                                     # installer compiler
```

**Build:**

```powershell
.\packaging\build.ps1                          # version 1.0.0
.\packaging\build.ps1 -Version 1.1.0           # set the version shown in Windows
.\packaging\build.ps1 -IncludeModel            # ship the model: works offline from first launch
```

It takes about 10 minutes and does four things:

1. Makes the Windows icon from `packaging\logo.png` (`packaging\make_icon.py`).
2. Bundles the app with PyInstaller (`packaging\facegen.spec`) into `dist\FaceGenerator\`.
3. Runs the bundled app's self-test (`FaceGenerator.exe --self-test <folder>`). This loads
   the model and generates one face with the age check, so a broken bundle stops the build.
4. Builds the installer with Inno Setup (`packaging\installer.iss`):
   `dist\installer\FaceGeneratorSetup-<version>.exe`. Give users this one file.

**What gets shipped.** Besides the code, the app ships the boundary files and your
`cache\age_regression.npz` if it exists, so the installed app has the same age calibration
as your checkout. The model (~325 MB) is downloaded on first launch, unless you build
with `-IncludeModel`. The InsightFace age-check model (`buffalo_l`) is always downloaded
on first use, into `%USERPROFILE%\.insightface`.

**Where the installed app keeps its files.** The install folder can be read-only, so the
installed app writes elsewhere:

| What | Where |
|---|---|
| Downloaded model, boundary files | `%LOCALAPPDATA%\FaceGenerator\cache\` |
| Log file | `%LOCALAPPDATA%\FaceGenerator\logs\facegen.log` |
| Generated images (default) | `%USERPROFILE%\FaceGenerator\output\` |

Uninstalling asks whether to delete the downloaded model and logs. Generated images are
always kept.

**Things to know:**
- The installed app takes about 1.3 GB, mostly TensorFlow, plus the model.
- It only runs on 64-bit Windows. A Mac or Linux version would have to be built on
  those systems.
- Windows SmartScreen will warn "Windows protected your PC" on other computers, because the
  installer isn't code-signed. Users click **More info > Run anyway**. Removing the
  warning needs a code-signing certificate.
- If you share the installer, the license terms below still apply. This is especially
  important with `-IncludeModel`, because the installer then contains NVIDIA's model:
  non-commercial use only, with credit.

---

## Troubleshooting

### Where to find logs
- The app writes `logs\facegen.log` in the project folder (rotated at 1 MB, 3 backups kept).
  The installed app writes `%LOCALAPPDATA%\FaceGenerator\logs\facegen.log` instead.
  It has timestamps, each generation run with a summary (images saved, time taken, how
  many attempts the age check rejected), every age-check retry, and full details of any error.
- Error dialogs show this path. Attach the file when reporting a problem.
- The console shows the same events without the per-retry detail.

### App will not start / import errors
- Make sure you activated `.venv37` before running `python app.py`.
- Confirm Python version is 3.7: `python --version`

### "Model loading..." never finishes
- Check your internet connection -- the model (~325 MB) must download on first run.
- An interrupted or failed download is discarded automatically (it never replaces the real file); just restart.

### "doesn't match the expected checksum"
- Every downloaded file is checked before use. This error means the file named in the
  message is corrupt, incomplete or not the expected file, so it wasn't loaded.
- Delete that file from `cache/` and restart the app to download it again.

### Images look psychedelic / corrupted
- This was caused by the age latent edit going out of distribution.
  It is fixed in the current version via delta clamping in `engine.py`.

### `fit_age_direction.py` crashes with `UnimplementedError: Depthwise convolution`
- Fixed in the current version -- the depthwise NCHW->NHWC patch is now included in the script.

### InsightFace `ValueError: operands could not be broadcast ... shapes (18,) (32,)`
- Was a `det_size=112` issue (not a multiple of 32). Fixed -- current version uses `det_size=160`.

### TensorFlow CUDA / GPU messages
TensorFlow's warnings and deprecation notices are hidden (errors still show), but one
line remains on machines without an NVIDIA GPU:
```
E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR (303)
```
This is harmless: TF 1.15 always looks for CUDA first. The app runs correctly on CPU.

### PowerShell execution policy error when activating venv
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## License

This project is built on NVIDIA's [StyleGAN](https://github.com/NVlabs/stylegan).
The NVIDIA material (the code in `dnnlib/`, `training/`, `metrics/` and the other files
carrying the NVIDIA copyright header, and the pretrained FFHQ model the app downloads) is
licensed under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/);
see [LICENSE.txt](LICENSE.txt).

- **Non-commercial use only.** For commercial use, see [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
- **Credit.** Cite the StyleGAN paper and indicate the changes you made:

  > **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
  > Tero Karras, Samuli Laine, Timo Aila (NVIDIA) — https://arxiv.org/abs/1812.04948

  This project adds a desktop UI (`app.py`, `components/`), latent age and gender editing
  (`engine.py`) and an age calibration script (`scripts/`) on top of the original code.

The age and gender boundary vectors (from [InterFaceGAN](https://github.com/genforce/interfacegan))
and the optional InsightFace models are third-party material under their own license terms.


