"""Extended tests for the hass module to increase coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hass import (
    cleanup_client,
    get_client,
    get_entities,
    get_hass_error_log,
    get_hass_version,
    get_system_overview,
    restart_home_assistant,
    summarize_domain,
)


class TestHassExtended:
    """Extended tests for Home Assistant API functions."""

    @pytest.mark.asyncio
    async def test_get_client_creation_and_reuse(self):
        """Test client creation and reuse."""
        # Clean up any existing client first
        await cleanup_client()

        # First call should create a new client
        client1 = await get_client()
        assert client1 is not None

        # Second call should reuse the same client
        client2 = await get_client()
        assert client1 is client2

        # Clean up
        await cleanup_client()

    @pytest.mark.asyncio
    async def test_cleanup_client(self):
        """Test client cleanup."""
        # Create a client first
        client = await get_client()
        assert client is not None

        # Mock the close method
        client.aclose = AsyncMock()

        # Clean up should close the client
        await cleanup_client()
        client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_hass_version(self, mock_config):
        """Test getting Home Assistant version."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"version": "2024.3.0"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    version = await get_hass_version()

                    assert version == "2024.3.0"
                    mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_home_assistant(self, mock_config):
        """Test restarting Home Assistant."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": "Restarting Home Assistant"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await restart_home_assistant()

                    assert "message" in result
                    mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entities_with_domain_filter(self, mock_config):
        """Test getting entities with domain filter."""
        mock_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "light.kitchen", "state": "off", "attributes": {}},
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_states

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    entities = await get_entities(domain="light")

                    assert isinstance(entities, list)
                    assert len(entities) == 2
                    assert all(e["entity_id"].startswith("light.") for e in entities)

    @pytest.mark.asyncio
    async def test_get_entities_with_search_query(self, mock_config):
        """Test getting entities with search query."""
        mock_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_states

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    entities = await get_entities(search_query="living")

                    assert isinstance(entities, list)
                    assert len(entities) == 1
                    assert "living" in entities[0]["entity_id"]

    @pytest.mark.asyncio
    async def test_get_entities_with_lean_format(self, mock_config):
        """Test getting entities with lean format."""
        mock_states = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {
                    "brightness": 255,
                    "friendly_name": "Living Room Light",
                    "extra_attr": "value",
                },
            },
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_states

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    entities = await get_entities(lean=True)

                    assert isinstance(entities, list)
                    assert len(entities) == 1
                    # Lean format should have limited attributes
                    assert "brightness" in entities[0]["attributes"]
                    assert "friendly_name" in entities[0]["attributes"]

    @pytest.mark.asyncio
    async def test_summarize_domain(self, mock_config):
        """Test domain summarization."""
        mock_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "light.kitchen", "state": "off", "attributes": {"brightness": 0}},
        ]

        with patch("app.hass.HA_URL", mock_config["hass_url"]):
            with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                with patch("app.hass.get_entities", return_value=mock_states):
                    summary = await summarize_domain("light")

                    assert summary["domain"] == "light"
                    assert summary["total_count"] == 2
                    assert "on" in summary["state_distribution"]
                    assert "off" in summary["state_distribution"]
                    assert len(summary["examples"]) > 0

    @pytest.mark.asyncio
    async def test_get_system_overview(self, mock_config):
        """Test system overview generation."""
        mock_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "switch.kitchen", "state": "off", "attributes": {}},
            {"entity_id": "sensor.temperature", "state": "22.5", "attributes": {"unit": "°C"}},
        ]

        # Mock the HTTP response for the direct API call in get_system_overview
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_states

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    overview = await get_system_overview()

                    assert "total_entities" in overview
                    assert "domains" in overview
                    assert "domain_samples" in overview
                    assert overview["total_entities"] == 3
                    assert "light" in overview["domains"]
                    assert "switch" in overview["domains"]
                    assert "sensor" in overview["domains"]

    @pytest.mark.asyncio
    async def test_get_hass_error_log_with_truncation(self, mock_config):
        """Test getting error log with truncation."""
        # Create a log that's actually longer than max_chars to trigger truncation
        large_log = "ERROR: Sample error message that is long enough to be truncated\n" * 100  # ~6000 chars

        # Mock the HTTP response directly for httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = large_log

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_error_log(max_chars=1000)

                    assert "log_text" in result
                    assert "error_count" in result
                    assert "truncated" in result
                    assert result["truncated"] is True
                    assert result["error_count"] > 0
                    assert len(result["log_text"]) <= 1200  # Account for truncation notice

    @pytest.mark.asyncio
    async def test_get_hass_error_log_with_tail_lines(self, mock_config):
        """Test getting error log with tail lines."""
        # Create a multi-line log for testing tail_lines
        full_log = "ERROR: Line 1\nWARNING: Line 2\nERROR: Line 3\nINFO: Line 4\nERROR: Line 5"

        # Mock the HTTP response directly for httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = full_log

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_error_log(tail_lines=2)

                    assert "log_text" in result
                    assert "error_count" in result
                    # Should contain last 2 lines
                    assert "Line 4" in result["log_text"]
                    assert "Line 5" in result["log_text"]
                    # Should NOT contain earlier lines
                    assert "Line 1" not in result["log_text"]
                    assert "Line 2" not in result["log_text"]

    @pytest.mark.asyncio
    async def test_error_handling_no_token(self):
        """Test error handling when no token is provided."""
        with patch("app.hass.HA_TOKEN", None):
            result = await get_hass_version()
            # The error handling decorator returns the error string directly for string return types
            assert isinstance(result, str)
            assert "token" in result.lower()

    @pytest.mark.asyncio
    async def test_error_handling_connection_error(self, mock_config):
        """Test error handling for connection errors."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert "error" in result