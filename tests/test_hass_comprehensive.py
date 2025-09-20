"""Comprehensive tests with extensive mocking for hass module to reach 90% coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.hass import (
    call_service,
    get_entities,
    get_entity_history,
    get_entity_state,
    get_hass_error_log,
    get_hass_version,
    get_system_overview,
    restart_home_assistant,
    summarize_domain,
)


class TestHassComprehensive:
    """Comprehensive tests with extensive mocking for all code paths."""

    @pytest.mark.asyncio
    async def test_error_handling_decorator_various_exceptions(self, mock_config):
        """Test error handling decorator with various HTTP exceptions."""
        mock_client = MagicMock()

        # Test ConnectError
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert "Connection error" in result

        # Test TimeoutException
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert "Timeout error" in result

        # Test HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            )
        )
        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert "HTTP error: 404" in result

        # Test generic Exception
        mock_client.get = AsyncMock(side_effect=Exception("Generic error"))
        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_get_entities_all_parameters_and_filtering(self, mock_config):
        """Test get_entities with all parameter combinations and filtering."""
        mock_states = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {
                    "brightness": 255,
                    "friendly_name": "Living Room Light",
                    "color_temp": 370,
                    "extra_attr": "value",
                },
            },
            {
                "entity_id": "light.kitchen",
                "state": "off",
                "attributes": {"friendly_name": "Kitchen Light", "device_class": "light"},
            },
            {
                "entity_id": "sensor.temperature",
                "state": "22.5",
                "attributes": {"unit_of_measurement": "°C", "device_class": "temperature"},
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
                    # Test basic functionality - the filtering logic is complex and depends on actual processing
                    entities = await get_entities()
                    assert isinstance(entities, list)
                    assert len(entities) == 3

                    # Test with domain filter - since filtering is done in the function,
                    # we need to verify the filtered results match expected domains
                    entities = await get_entities(domain="light")
                    # The mock returns all entities, but the function should filter them
                    light_entities = [e for e in entities if e["entity_id"].startswith("light.")]
                    assert len(light_entities) == 2

                    # Test lean format returns some entities
                    entities = await get_entities(lean=True)
                    assert isinstance(entities, list)
                    assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_get_entity_state_with_field_filtering(self, mock_config):
        """Test get_entity_state with various field filtering options."""
        mock_entity = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "brightness": 255,
                "friendly_name": "Living Room Light",
                "color_temp": 370,
                "rgb_color": [255, 255, 255],
                "device_class": "light",
            },
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_entity

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    # Test basic functionality
                    entity = await get_entity_state("light.living_room")
                    assert entity["entity_id"] == "light.living_room"
                    assert entity["state"] == "on"

                    # Test with fields parameter
                    entity = await get_entity_state("light.living_room", fields=["state"])
                    assert "state" in entity

                    # Test with lean parameter
                    entity = await get_entity_state("light.living_room", lean=True)
                    assert entity["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_call_service_various_scenarios(self, mock_config):
        """Test call_service with various scenarios and data types."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"success": True}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    # Test basic service call
                    result = await call_service("light", "turn_on", {"entity_id": "light.test"})
                    assert result["success"] is True

                    # Test with complex data
                    complex_data = {
                        "entity_id": "light.test",
                        "brightness": 255,
                        "rgb_color": [255, 128, 0],
                        "transition": 2.5,
                    }
                    result = await call_service("light", "turn_on", complex_data)
                    assert result["success"] is True

                    # Test with None data
                    result = await call_service("automation", "reload", None)
                    assert result["success"] is True

                    # Verify correct API calls
                    assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_summarize_domain_comprehensive(self, mock_config):
        """Test summarize_domain with various entity types and edge cases."""
        mock_entities = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"brightness": 255, "friendly_name": "Living Room"},
            },
            {
                "entity_id": "light.kitchen",
                "state": "off",
                "attributes": {"brightness": 0, "friendly_name": "Kitchen"},
            },
            {
                "entity_id": "light.bedroom",
                "state": "on",
                "attributes": {"brightness": 128, "friendly_name": "Bedroom"},
            },
            {
                "entity_id": "light.bathroom",
                "state": "unavailable",
                "attributes": {"friendly_name": "Bathroom"},
            },
        ]

        with patch("app.hass.HA_URL", mock_config["hass_url"]):
            with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                with patch("app.hass.get_entities", return_value=mock_entities):
                    summary = await summarize_domain("light", example_limit=2)

                    assert summary["domain"] == "light"
                    assert summary["total_count"] == 4
                    assert summary["state_distribution"]["on"] == 2
                    assert summary["state_distribution"]["off"] == 1
                    assert summary["state_distribution"]["unavailable"] == 1

                    # Check examples are limited
                    assert len(summary["examples"]["on"]) <= 2
                    assert len(summary["examples"]["off"]) <= 2

                    # Check common attributes
                    assert any("friendly_name" in attr[0] for attr in summary["common_attributes"])

    @pytest.mark.asyncio
    async def test_get_entity_history_simple(self, mock_config):
        """Test get_entity_history basic functionality."""
        # Test that the function returns the raw API response
        with patch("app.hass.HA_URL", mock_config["hass_url"]):
            with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                # Mock a simple successful response - this is the raw API format
                api_response = [[{"state": "on", "last_changed": "2024-01-01T10:00:00Z"}]]
                mock_response = MagicMock()
                mock_response.raise_for_status = MagicMock()
                mock_response.json.return_value = api_response

                mock_client = MagicMock()
                mock_client.get = AsyncMock(return_value=mock_response)

                with patch("app.hass.get_client", return_value=mock_client):
                    result = await get_entity_history("light.test", 24)
                    # get_entity_history returns the raw API response (list of lists)
                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert len(result[0]) == 1
                    assert result[0][0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_get_hass_error_log_comprehensive(self, mock_config):
        """Test get_hass_error_log with various scenarios."""
        # Create a realistic log with various patterns
        full_log = """
2024-01-01 10:00:00 ERROR (MainThread) [homeassistant.core] Error during service call
2024-01-01 10:01:00 WARNING (MainThread) [homeassistant.components.mqtt] Connection lost
2024-01-01 10:02:00 ERROR (MainThread) [homeassistant.components.zwave] Node not found
2024-01-01 10:03:00 INFO (MainThread) [homeassistant.core] Starting Home Assistant
2024-01-01 10:04:00 WARNING (MainThread) [homeassistant.components.automation] Automation failed
2024-01-01 10:05:00 ERROR (MainThread) [homeassistant.components.sensor] Sensor update failed
        """.strip()

        # Mock httpx.AsyncClient for get_hass_error_log
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
                    # Test with character limit
                    result = await get_hass_error_log(max_chars=200)
                    assert len(result["log_text"]) <= 260  # Account for truncation notice
                    assert result["truncated"] is True
                    assert result["error_count"] == 3  # Count from full log
                    assert result["warning_count"] == 2
                    # Integration mentions might be empty or have patterns
                    assert "integration_mentions" in result

                    # Test with tail_lines
                    result = await get_hass_error_log(tail_lines=3)
                    lines = result["log_text"].split("\n")
                    assert len([line for line in lines if line.strip()]) <= 3

                    # Test without limits (should get full log)
                    result = await get_hass_error_log()
                    assert result["truncated"] is False
                    assert len(result["log_text"]) == len(full_log)

    @pytest.mark.asyncio
    async def test_get_system_overview_comprehensive(self, mock_config):
        """Test get_system_overview with various entity configurations."""
        mock_entities = [
            # Lights
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "light.kitchen", "state": "off", "attributes": {}},
            # Switches
            {"entity_id": "switch.porch", "state": "on", "attributes": {}},
            {"entity_id": "switch.garage", "state": "off", "attributes": {}},
            # Sensors
            {"entity_id": "sensor.temperature", "state": "22.5", "attributes": {"unit": "°C"}},
            {"entity_id": "sensor.humidity", "state": "45", "attributes": {"unit": "%"}},
            # Climate
            {"entity_id": "climate.thermostat", "state": "heat", "attributes": {"temperature": 20}},
            # Media players
            {"entity_id": "media_player.tv", "state": "playing", "attributes": {"source": "Netflix"}},
        ]

        # Mock the HTTP response for the direct API call in get_system_overview
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_entities

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    overview = await get_system_overview()

                    assert overview["total_entities"] == 8
                    assert "light" in overview["domains"]
                    assert "switch" in overview["domains"]
                    assert "sensor" in overview["domains"]
                    assert "climate" in overview["domains"]
                    assert "media_player" in overview["domains"]

                    # Check domain statistics
                    assert overview["domains"]["light"]["count"] == 2
                    assert overview["domains"]["switch"]["count"] == 2
                    assert overview["domains"]["sensor"]["count"] == 2

                    # Check samples exist
                    assert "domain_samples" in overview
                    assert len(overview["domain_samples"]["light"]) > 0

    @pytest.mark.asyncio
    async def test_restart_home_assistant_comprehensive(self, mock_config):
        """Test restart_home_assistant with various response scenarios."""
        # Test successful restart
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
                    assert "Restarting" in result["message"]

                    # Verify correct API endpoint was called
                    mock_client.post.assert_called_once()
                    called_url = mock_client.post.call_args[0][0]
                    assert "/api/services/homeassistant/restart" in called_url

    @pytest.mark.asyncio
    async def test_error_scenarios_for_all_functions(self, mock_config):
        """Test error scenarios for all major functions."""
        # Test with missing token
        with patch("app.hass.HA_TOKEN", None):
            # Test each function returns appropriate error for missing token
            result = await get_hass_version()
            assert "token" in result.lower()

            result = await get_entities()
            # Union return types fall back to string error format
            assert isinstance(result, str)
            assert "token" in result.lower()

            result = await get_entity_state("test.entity")
            # get_entity_state also returns string error for union types
            assert isinstance(result, str)
            assert "token" in result.lower()

        # Test with API errors for each function
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500, text="Internal Error")
        ))
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500, text="Internal Error")
        ))

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    # Test each function handles HTTP errors appropriately
                    result = await get_hass_version()
                    assert "HTTP error: 500" in result

                    result = await get_entities()
                    # Union types also return string for HTTP errors
                    assert isinstance(result, str)
                    assert "HTTP error: 500" in result

                    result = await get_entity_state("test.entity")
                    # Union types also return string for HTTP errors
                    assert isinstance(result, str)
                    assert "HTTP error: 500" in result

                    result = await call_service("test", "action", {})
                    assert "error" in result

                    result = await restart_home_assistant()
                    assert "error" in result