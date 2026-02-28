"""Pytest configuration — register custom markers."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that run the full pipeline on local data"
    )
