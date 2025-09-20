"""Entry point for running Hass-MCP via uv/uvx tool"""

from app.server import mcp


def main() -> None:
    """Run the MCP server with stdio communication"""
    mcp.run()
