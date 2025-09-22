"""Basic tests to ensure the setup is working."""


def test_import():
    """Test that we can import the data_agent module."""
    import data_agent

    assert data_agent.__version__ == "0.1.0"


def test_basic_assertion():
    """Basic test to ensure pytest is working."""
    assert 1 + 1 == 2
