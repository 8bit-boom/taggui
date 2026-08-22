from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QIcon, QImageReader, QPixmap


@dataclass
class Image:
    path: Path
    dimensions: tuple[int, int] | None
    tags: list[str] = field(default_factory=list)
    thumbnail: QIcon | None = None


def load_pixmap(image_path: Path) -> QPixmap:
    """
    Load an image as a `QPixmap`, applying its Exif orientation. Qt's
    `QImageReader` is tried first because it is faster; if it fails to
    produce an image (for example for formats Qt does not natively support,
    such as AVIF), fall back to loading the image with Pillow.
    """
    image_reader = QImageReader(str(image_path))
    image_reader.setAutoTransform(True)
    pixmap = QPixmap.fromImageReader(image_reader)
    if not pixmap.isNull():
        return pixmap
    with PilImage.open(image_path) as pil_image:
        pil_image = exif_transpose(pil_image)
        pil_image = pil_image.convert('RGBA')
        qimage = ImageQt(pil_image)
        pixmap = QPixmap.fromImage(qimage)
    return pixmap


def get_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """
    Get the dimensions of an image using Pillow. This is used as a fallback
    for formats that `imagesize` cannot read the dimensions of, such as
    AVIF.
    """
    try:
        with PilImage.open(image_path) as pil_image:
            pil_image = exif_transpose(pil_image)
            return pil_image.size
    except OSError:
        return None
