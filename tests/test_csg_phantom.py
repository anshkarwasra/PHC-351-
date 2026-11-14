import numpy as np

from src.synthetic.csg_phantom import csg_density


def test_csg_phantom_is_repeatable_and_has_thin_structure():
    params = {"body_scale": 1.0, "rib_thickness": 0.04, "channel_radius": 0.04}

    first = csg_density(48, params)
    second = csg_density(48, params)

    assert first.dtype == np.float32
    assert first.shape == (48, 48, 48)
    assert np.array_equal(first, second)
    assert 0 < first.mean() < 0.5
