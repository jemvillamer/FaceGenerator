"""Make the Windows icon (packaging/facegen.ico) from the app logo
(packaging/logo.png). Run by build.ps1 before PyInstaller; the .exe and the
installer use the .ico, the running app uses logo.png directly.

    python packaging/make_icon.py
"""
import os

import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "logo.png")
OUT = os.path.join(HERE, "facegen.ico")
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


def main():
    logo = PIL.Image.open(LOGO).convert("RGBA")
    # Pillow resizes the largest image to each size in `sizes`.
    logo.resize((256, 256), PIL.Image.LANCZOS).save(OUT, sizes=[(s, s) for s in SIZES])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
