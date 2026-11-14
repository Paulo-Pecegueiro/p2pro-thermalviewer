'''
Orientation of the thermal image: rotation and mirroring, for a camera mounted sideways or upside down.

The sensor image (H x W = 192 x 256) is first rotated clockwise by "rotate" degrees (0, 90, 180, 270), then
mirrored: "flip_h" swaps left and right, "flip_v" top and bottom of the rotated image. With 90 or 270 the image
is 192 wide and 256 high. Everything the server sends (image, positions, ROIs) is in the oriented image.

Coordinates here are continuous: pixel (i, j) covers [i, i+1) x [j, j+1).
'''
import math

ROTATIONS = (0, 90, 180, 270)


class OrientationError(ValueError):
    'invalid orientation (HTTP 400)'


class Orientation:
    'immutable'

    def __init__(self, rotate=0, flip_h=False, flip_v=False):
        self.rotate = rotate
        self.flip_h = flip_h
        self.flip_v = flip_v

    @classmethod
    def from_dict(cls, d, base=None):
        '''validates {"rotate": 0|90|180|270, "flip_h": bool, "flip_v": bool}. Missing keys are taken from base
        (default: no rotation, no mirroring). rotate may be any multiple of 90 (-90 -> 270, 450 -> 90).'''
        if not isinstance(d, dict):
            raise OrientationError('orientation must be a JSON object')
        unknown = set(d) - {'rotate', 'flip_h', 'flip_v'}
        if unknown:
            raise OrientationError(f'unknown orientation keys {sorted(unknown)} (allowed: rotate, flip_h, flip_v)')
        cur = (base or cls()).to_dict() | d
        rot = cur['rotate']
        if isinstance(rot, bool) or not isinstance(rot, (int, float)) or not math.isfinite(rot) or rot % 90:
            raise OrientationError('"rotate" must be a multiple of 90 (0, 90, 180, 270)')
        for k in ('flip_h', 'flip_v'):
            if not isinstance(cur[k], bool):
                raise OrientationError(f'"{k}" must be true or false')
        return cls(int(rot) % 360, cur['flip_h'], cur['flip_v'])

    def to_dict(self):
        return {'rotate': self.rotate, 'flip_h': self.flip_h, 'flip_v': self.flip_v}

    def __eq__(self, other):
        return isinstance(other, Orientation) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return f'Orientation({self.rotate}, flip_h={self.flip_h}, flip_v={self.flip_v})'

    @property
    def swaps_axes(self):
        return self.rotate in (90, 270)

    def shape(self, sensor_shape):
        '(rows, cols) of the oriented image'
        h, w = sensor_shape
        return (w, h) if self.swaps_axes else (h, w)

    def apply(self, img):
        'oriented view of a sensor image (rows x cols [x ...]), no copy'
        if self.rotate:
            img = img.swapaxes(0, 1)[:, ::-1] if self.rotate == 90 else \
                img[::-1, ::-1] if self.rotate == 180 else img.swapaxes(0, 1)[::-1]
        if self.flip_h:
            img = img[:, ::-1]
        if self.flip_v:
            img = img[::-1]
        return img

    def from_sensor(self, x, y, sensor_shape):
        'sensor image point -> oriented image point'
        h, w = sensor_shape
        if self.rotate == 90:
            x, y = h - y, x
        elif self.rotate == 180:
            x, y = w - x, h - y
        elif self.rotate == 270:
            x, y = y, w - x
        oh, ow = self.shape(sensor_shape)
        if self.flip_h:
            x = ow - x
        if self.flip_v:
            y = oh - y
        return x, y

    def to_sensor(self, x, y, sensor_shape):
        'oriented image point -> sensor image point'
        h, w = sensor_shape
        oh, ow = self.shape(sensor_shape)
        if self.flip_h:
            x = ow - x
        if self.flip_v:
            y = oh - y
        if self.rotate == 90:
            x, y = y, h - x
        elif self.rotate == 180:
            x, y = w - x, h - y
        elif self.rotate == 270:
            x, y = w - y, x
        return x, y


def remap_roi(d, old, new, sensor_shape):
    '''ROI definition (see roi.py) in the image oriented by old -> the same sensor pixels in the image oriented
    by new. Only the geometry changes.'''
    def move(x, y):
        return new.from_sensor(*old.to_sensor(x, y, sensor_shape), sensor_shape)

    d = dict(d)
    if d['type'] == 'point':
        x, y = move(d['x'] + 0.5, d['y'] + 0.5)  # pixel center
        d['x'], d['y'] = math.floor(x), math.floor(y)
    elif d['type'] == 'rect':
        x0, y0 = move(d['x'], d['y'])
        x1, y1 = move(d['x'] + d['w'], d['y'] + d['h'])
        d.update(x=round(min(x0, x1), 2), y=round(min(y0, y1), 2), w=round(abs(x1 - x0), 2), h=round(abs(y1 - y0), 2))
    else:
        cx, cy = move(d['cx'], d['cy'])
        d.update(cx=round(cx, 2), cy=round(cy, 2))
    return d
