"""
read_header_resolution must predict LoadImage's decoded size exactly, or
decline. LoadImage decodes with torchvision (EXIF orientation applied to JPEG
and PNG only) and falls back to Pillow's exif_transpose for anything
torchvision rejects; these tests decode each sample the same way.
"""

from mgds.pipelineModules.SmartDiskCache import EXIF_ORIENTATION_TAG, read_header_resolution

import torch
from torchvision.io import ImageReadMode, read_image

import pytest
from PIL import Image, ImageOps

WIDTH, HEIGHT = 48, 32


def _decoded_size(path: str) -> tuple[int, int]:
    """(height, width) as LoadImage.get_item decodes the file."""
    try:
        tensor = read_image(path, ImageReadMode.RGB, apply_exif_orientation=True)
    except RuntimeError:
        with Image.open(path) as image:
            tensor = torch.zeros(ImageOps.exif_transpose(image).convert("RGB").size[::-1])
        return tuple(tensor.shape)
    return tuple(tensor.shape[1:])


def _save(tmp_path, name: str, orientation: int | None = None, **kwargs) -> str:
    path = str(tmp_path / name)
    image = Image.new("RGB", (WIDTH, HEIGHT), color=(200, 30, 30))
    if orientation is not None:
        exif = Image.Exif()
        exif[EXIF_ORIENTATION_TAG] = orientation
        kwargs["exif"] = exif.tobytes()
    image.save(path, **kwargs)
    return path


@pytest.mark.parametrize("orientation", [None, 1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("extension", [".jpg", ".png"])
def test_exif_oriented_formats_match_decode(tmp_path, extension, orientation):
    path = _save(tmp_path, f"image{extension}", orientation)

    header = read_header_resolution(path)

    assert header is not None
    assert header == _decoded_size(path)


def test_webp_without_rotation_matches_decode(tmp_path):
    path = _save(tmp_path, "image.webp", lossless=True)

    assert read_header_resolution(path) == _decoded_size(path) == (HEIGHT, WIDTH)


def test_webp_with_rotation_declines(tmp_path):
    # torchvision ignores WebP EXIF, but its Pillow fallback would honour it.
    path = _save(tmp_path, "image.webp", orientation=6, lossless=True)

    assert read_header_resolution(path) is None


@pytest.mark.parametrize("extension", [".bmp", ".tif"])
def test_other_formats_decline(tmp_path, extension):
    assert read_header_resolution(_save(tmp_path, f"image{extension}")) is None


def test_animated_gif_declines(tmp_path):
    path = str(tmp_path / "anim.gif")
    frames = [Image.new("RGB", (WIDTH, HEIGHT), color=(i * 80, 0, 0)) for i in range(3)]
    frames[0].save(path, save_all=True, append_images=frames[1:])

    assert read_header_resolution(path) is None


def test_unreadable_file_declines(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")

    assert read_header_resolution(str(path)) is None
    assert read_header_resolution(str(tmp_path / "missing.png")) is None
