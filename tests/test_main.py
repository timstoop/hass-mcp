"""Tests for the main entry point module."""

from unittest.mock import patch


class TestMain:
    """Test the main entry point functionality."""

    def test_main_entry_point(self):
        """Test that main function calls mcp.run()."""
        with patch("app.__main__.mcp") as mock_mcp:
            from app.__main__ import main

            main()
            mock_mcp.run.assert_called_once()

    def test_main_module_execution(self):
        """Test that main module can be imported and executed."""
        with patch("app.__main__.mcp") as mock_mcp:
            # Import the module to test __name__ == "__main__" path
            exec(
                'from app.__main__ import main; main()',
                {"__name__": "__main__"},
            )
            mock_mcp.run.assert_called_once()