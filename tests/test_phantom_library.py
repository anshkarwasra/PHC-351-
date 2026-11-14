import pytest

from src.synthetic.phantom_library import PhantomLibrary


def test_missing_config_raises_file_not_found(tmp_path):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        PhantomLibrary(config_or_path=missing_config)


def test_explicit_config_dict_is_supported(tmp_path):
    library = PhantomLibrary(
        config_or_path={
            "grid": {"n_clean": 2, "n_deg": 2},
            "geometry": {"kind": "shepp_logan"},
        },
        out_dir=tmp_path,
    )

    assert library.n_clean == 2
    assert library.n_deg == 2
