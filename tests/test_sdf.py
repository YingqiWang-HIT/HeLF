import numpy as np

from helf.data.sdf import polygon_signed_distance


def test_polygon_signed_distance_sign_and_shape():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    sdf, x, y = polygon_signed_distance(square, 32, 32, xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    assert sdf.shape == (32, 32)
    assert x.shape == y.shape == sdf.shape
    center = sdf[16, 16]
    corner = sdf[0, 0]
    assert center < 0
    assert corner > 0
