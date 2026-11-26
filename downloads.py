"""Downloaded files and their checksums, plus helpers that verify them before use.

Standard library only (no TensorFlow/Qt), so both the app (engine.py) and
scripts/fit_age_direction.py can import it.

Reference checksums were confirmed against independent sources:
  - the model matches NVIDIA's own copy on nvlabs-fi-cdn.nvidia.com
    (same size and S3 multipart ETag 0c2555a45b54bed665a2505ff9de6412-39);
  - the boundaries match InterFaceGAN's files at INTERFACEGAN_COMMIT, which
    are identical to the original 2019 release.
"""
import hashlib
import os

# ── Pretrained StyleGAN generator (FFHQ 1024x1024) ─────────────────────────
MODEL_FILE = 'karras2019stylegan-ffhq-1024x1024.pkl'
MODEL_GDRIVE_ID = '1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
MODEL_SHA256 = 'f277a43711447812d809b2fdd7748b945cfd7bdcd391d66e557a68d590309aab'

# ── InterFaceGAN latent boundaries (pinned: master has changed before) ──────
INTERFACEGAN_COMMIT = '8da3fc0fe2a1d4c88dc5f9bee65e8077093ad2bb'
BOUNDARY_SHA256 = {
    'stylegan_ffhq_age_boundary.npy':
        '4317407b6601233af2ccdd912f30622636309a05c799c36cc5ce25271bb66488',
    'stylegan_ffhq_gender_boundary.npy':
        '4fd66591b88ab174605db9e8e7fb2c2cdcd366d0bf4d068cbee0843e04c02100',
}


def boundary_url(filename):
    return (f'https://raw.githubusercontent.com/genforce/interfacegan/'
            f'{INTERFACEGAN_COMMIT}/boundaries/{filename}')


class ChecksumError(Exception):
    """A file doesn't match its expected checksum. The message is user-facing."""


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def verify_existing(path, sha256):
    """Raise ChecksumError if the file at `path` isn't the expected one."""
    if sha256_of(path) != sha256:
        raise ChecksumError(
            f"{path}\n\ndoesn't match the expected checksum: it may be corrupt, "
            "incomplete or a different file, so it wasn't loaded.\n\n"
            "Delete it and restart the app to download it again.")


def fetch_verified(download, dest, sha256):
    """Download to `dest` safely.

    `download(tmp_path)` must write the file to tmp_path. The result is checked
    and only then renamed to `dest`, so an interrupted or wrong download never
    leaves a file under the real name.
    """
    part = dest + '.part'
    if os.path.exists(part):
        os.remove(part)  # left over from an earlier interrupted attempt
    try:
        download(part)
        if not os.path.isfile(part):
            raise RuntimeError(f'Download of {os.path.basename(dest)} failed '
                               '(no file was written).')
        if sha256_of(part) != sha256:
            raise ChecksumError(
                f"The downloaded {os.path.basename(dest)} doesn't match the "
                "expected checksum, so it wasn't used.\n\nThe download may have "
                "been cut off, or the server returned something else (for "
                "example an error page). Please try again later.")
        os.replace(part, dest)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
