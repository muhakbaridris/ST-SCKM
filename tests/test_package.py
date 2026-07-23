from stsckm import STSCKM, __version__


def test_public_imports():
    assert STSCKM.__name__ == "STSCKM"
    assert __version__ == "1.0.0"
