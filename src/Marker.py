from enum import IntEnum, unique
from src.constants import IMAGES_DIR
import os


@unique
class Marker(IntEnum):
    RIGHT = 1
    LEFT = 2
    IDLE = 3
    TOUNGE = 4

    @property
    def image_path(self):
        return os.path.join(IMAGES_DIR, f'{self.name}.png')

    @property
    def get_ready_text(self):
        if self == Marker.LEFT:
            return "get ready for LEFT!"
        if self == Marker.RIGHT:
            return "get ready for RIGHT"
        if self == Marker.IDLE:
            return "get ready for IDLE"

    @property
    def shape(self):
        if self == Marker.LEFT or self == Marker.RIGHT:
            return create_arrow_shape(self)
        else:
            return create_square_shape()

    @property
    def what_to_show(self):
        if self == Marker.TOUNGE:
            return "image"
        return "shape"

    @classmethod
    def all(cls):
        return [stim.value for stim in cls]


def create_arrow_shape(marker=Marker.RIGHT):
    arrowVert = [(-0.4, 0.05), (-0.4, -0.05), (-.2, -0.05), (-.2, -0.1), (0, 0), (-.2, 0.1), (-.2, 0.05)]
    width = 1
    height = 2
    scale = 2.5
    if marker == Marker.LEFT:
        scale *= -1
    arrowVert = [(scale * width * x, scale * height * y) for (x, y) in arrowVert]
    center = tuple(map(sum, zip(*arrowVert)))
    center = (-center[0] / len(arrowVert), -center[1] / len(arrowVert))
    arrowVert = [tuple(map(sum, zip(p, center))) for p in arrowVert]
    return arrowVert


def create_square_shape():
    width = 1
    height = 1.5
    scale = 0.6
    Vert = [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]
    Vert = [(scale * width * x, scale * height * y) for (x, y) in Vert]
    return Vert


create_arrow_shape()
