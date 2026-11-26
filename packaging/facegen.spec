# PyInstaller spec for the desktop app. Build with packaging/build.ps1, which
# runs:  python -m PyInstaller --noconfirm --distpath dist --workpath build packaging/facegen.spec
#
# Output: dist/FaceGenerator/FaceGenerator.exe plus its libraries ("one-folder"
# mode; one-file mode would unpack ~1 GB of TensorFlow on every launch).
#
# Data files shipped in the app (read-only; see config.data_file):
#   - the InterFaceGAN boundaries and cache/age_regression.npz (if present),
#     so the app has the same age calibration as this checkout;
#   - the StyleGAN model, only when FACEGEN_INCLUDE_MODEL=1 (build.ps1
#     -IncludeModel). Otherwise the app downloads it (~325 MB) on first launch.
import os

ROOT = os.path.dirname(SPECPATH)           # SPECPATH: the folder of this file
CACHE = os.path.join(ROOT, 'cache')

MODEL_FILE = 'karras2019stylegan-ffhq-1024x1024.pkl'
BUNDLED_DATA = ['stylegan_ffhq_age_boundary.npy',
                'stylegan_ffhq_gender_boundary.npy',
                'age_regression.npz']
if os.environ.get('FACEGEN_INCLUDE_MODEL') == '1':
    if not os.path.isfile(os.path.join(CACHE, MODEL_FILE)):
        raise SystemExit(f'-IncludeModel: {MODEL_FILE} is not in {CACHE}. '
                         'Run the app from source once to download it.')
    BUNDLED_DATA.append(MODEL_FILE)

datas = [
    (os.path.join(ROOT, 'style.qss'), '.'),
    (os.path.join(SPECPATH, 'logo.png'), 'packaging'),
]
for name in BUNDLED_DATA:
    path = os.path.join(CACHE, name)
    if os.path.isfile(path):
        datas.append((path, 'data'))
        print(f'[facegen.spec] shipping {name}')
    else:
        print(f'[facegen.spec] not shipping {name} (not in cache/)')

a = Analysis(
    [os.path.join(ROOT, 'app.py')],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=[
        # The model pickle rebuilds its network from source code stored in the
        # file; that code imports these at load time, which PyInstaller can't see.
        'dnnlib', 'dnnlib.tflib', 'dnnlib.tflib.network', 'dnnlib.tflib.tfutil',
        'dnnlib.tflib.autosummary', 'dnnlib.tflib.optimizer', 'dnnlib.util',
        # Optional age check (generator.AgeEstimator imports it lazily).
        'insightface', 'insightface.app', 'cv2',
    ],
    excludes=['tkinter', 'pytest', 'IPython', 'notebook', 'jupyter'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceGenerator',
    icon=os.path.join(SPECPATH, 'facegen.ico'),
    console=False,          # no console window; logs go to the log file
    upx=False,              # UPX can break TensorFlow's DLLs
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name='FaceGenerator',
)
