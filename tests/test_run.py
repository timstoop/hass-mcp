"""Tests for the run module."""

from unittest.mock import patch


class TestRun:
    """Test the run module functionality."""

    def test_run_main_function(self):
        """Test that main function in run module calls mcp.run()."""
        with patch("app.run.mcp") as mock_mcp:
            from app.run import main

            main()
            mock_mcp.run.assert_called_once()

    def test_run_module_execution(self):
        """Test that run module can be executed directly."""
        with patch("app.run.mcp") as mock_mcp:
            # Test the module execution path
            exec(
                'from app.run import main; main()',
                {"__name__": "__main__"},
            )
            mock_mcp.run.assert_called_once()