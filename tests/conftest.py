"""
Pytest configuration for test isolation.
Ensures that module mocking in one test file doesn't affect others.

IMPORTANT: backend_test.py (now z_backend_test.py) mocks backend.src.rag_service
at module load time. To ensure test_rag_service.py gets the real module,
backend_test has been renamed to z_backend_test.py to load alphabetically last.
"""
import sys


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before collection.
    Pre-import the real rag_service module before z_backend_test.py is loaded.
    """
    # Add project root to path
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Pre-import the real module - this happens before test file collection
    # which loads z_backend_test.py and mocks it
    try:
        import backend.src.rag_service  # noqa: F401
    except ImportError:
        pass  # May fail in CI if dependencies aren't installed yet



