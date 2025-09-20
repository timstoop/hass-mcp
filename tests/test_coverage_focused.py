"""Focused tests to target specific missing coverage areas."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCoverageFocused:
    """Tests focused on specific lines/branches to increase coverage."""

    @pytest.mark.asyncio
    async def test_server_edge_cases(self):
        """Test edge cases in server functions."""
        from app.server import (
            get_entity,
            list_entities,
            search_entities_tool,
        )

        # Test error handling paths in server functions
        error_response = {"error": "Test error"}

        # Test get_entity with error response - mock at server level since it's imported
        with patch("app.server.get_entity_state", return_value=error_response):
            result = await get_entity("test.entity")
            assert result == error_response

        # Test list_entities with error response - should return empty list
        with patch("app.server.get_entities", return_value=error_response):
            result = await list_entities()
            assert result == []

        # Test search_entities_tool with error response
        with patch("app.server.get_entities", return_value=error_response):
            result = await search_entities_tool("test")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_hass_edge_cases(self, mock_config):
        """Test edge cases in hass functions."""
        from app.hass import get_entities, get_entity_state

        # Test various parameter combinations to hit different branches
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = []

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    # Test empty results
                    result = await get_entities()
                    assert result == []

                    # Test with various combinations to hit branches
                    await get_entities(domain="test", limit=50)
                    await get_entities(search_query="test", lean=False)
                    await get_entities(fields=["state"])

        # Test get_entity_state edge cases
        mock_response.json.return_value = {"entity_id": "test", "state": "on", "attributes": {}}
        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    await get_entity_state("test", fields=["state"])
                    await get_entity_state("test", lean=False)

    @pytest.mark.asyncio
    async def test_prompt_edge_cases(self):
        """Test prompt functions with various inputs."""
        from app.server import create_automation, debug_automation, troubleshoot_entity

        # Test different trigger types for create_automation
        trigger_types = ["time", "state", "device", "webhook"]
        for trigger_type in trigger_types:
            result = create_automation(trigger_type)
            assert isinstance(result, list)

        # Test debug_automation with different automation IDs
        result = debug_automation("test.automation")
        assert isinstance(result, list)

        # Test troubleshoot_entity with different entity IDs
        result = troubleshoot_entity("test.entity")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_domain_handling_edge_cases(self, mock_config):
        """Test domain-specific handling edge cases."""
        from app.hass import summarize_domain

        # Test with empty domain results
        with patch("app.hass.HA_URL", mock_config["hass_url"]):
            with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                with patch("app.hass.get_entities", return_value=[]):
                    result = await summarize_domain("empty_domain")
                    assert result["total_count"] == 0

                # Test with single entity
                single_entity = [{"entity_id": "light.test", "state": "on", "attributes": {}}]
                with patch("app.hass.get_entities", return_value=single_entity):
                    result = await summarize_domain("light", example_limit=1)
                    assert result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_resource_functions_edge_cases(self):
        """Test resource functions with edge cases."""
        from app.server import (
            get_entity_resource,
            list_states_by_domain_resource,
            search_entities_resource_with_limit,
        )

        # Test with error responses - mock at server level
        with patch("app.server.get_entity_state", return_value={"error": "Not found"}):
            result = await get_entity_resource("nonexistent.entity")
            assert "error" in result.lower() or "Error" in result

        # Test with empty results - mock at server level
        with patch("app.server.get_entities", return_value=[]):
            result = await list_states_by_domain_resource("empty")
            assert "Empty Entities" in result or "empty" in result

            result = await search_entities_resource_with_limit("nothing", "10")
            assert "nothing" in result

    @pytest.mark.asyncio
    async def test_automation_edge_cases(self):
        """Test automation-related edge cases."""
        from app.server import list_automations

        # Test with various error formats
        error_formats = [
            {"error": "404 Not Found"},
            [{"error": "Connection failed"}],
            Exception("Network error"),
        ]

        for error_format in error_formats:
            if isinstance(error_format, Exception):
                with patch("app.server.get_automations", side_effect=error_format):
                    result = await list_automations()
                    assert result == []
            else:
                with patch("app.server.get_automations", return_value=error_format):
                    result = await list_automations()
                    assert result == []

    @pytest.mark.asyncio
    async def test_service_call_edge_cases(self, mock_config):
        """Test service call edge cases."""
        from app.hass import call_service

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": "success"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    # Test with None data
                    result = await call_service("test", "action", None)
                    assert "result" in result

                    # Test with empty data
                    result = await call_service("test", "action", {})
                    assert "result" in result

    @pytest.mark.asyncio
    async def test_entity_action_edge_cases(self):
        """Test entity_action with various entity types."""
        from app.server import entity_action

        # Test with different entity domains
        entity_types = [
            "light.test",
            "switch.test",
            "fan.test",
            "cover.test",
            "climate.test",
            "media_player.test",
        ]

        mock_result = {"success": True}

        for entity_id in entity_types:
            with patch("app.server.call_service", return_value=mock_result):
                result = await entity_action(entity_id, "on")
                assert result == mock_result

                result = await entity_action(entity_id, "off")
                assert result == mock_result

                result = await entity_action(entity_id, "toggle")
                assert result == mock_result

    @pytest.mark.asyncio
    async def test_system_functions_edge_cases(self, mock_config):
        """Test system-level functions edge cases."""
        from app.hass import get_hass_version, restart_home_assistant

        # Test version retrieval edge cases
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        # Test with missing version in response
        mock_response.json.return_value = {}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await get_hass_version()
                    assert result == "unknown"

        # Test restart with different response formats
        mock_response.json.return_value = {"message": "Restarting"}
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.hass.get_client", return_value=mock_client):
            with patch("app.hass.HA_URL", mock_config["hass_url"]):
                with patch("app.hass.HA_TOKEN", mock_config["hass_token"]):
                    result = await restart_home_assistant()
                    assert "message" in result

    @pytest.mark.asyncio
    async def test_client_management_edge_cases(self):
        """Test client management edge cases."""
        from app.hass import cleanup_client, get_client

        # Test cleanup when client is None
        await cleanup_client()

        # Test client creation
        client = await get_client()
        assert client is not None

        # Test cleanup with actual client
        await cleanup_client()

    @pytest.mark.asyncio
    async def test_search_logic_edge_cases(self):
        """Test search logic edge cases."""
        from app.server import search_entities_tool

        # Test with empty query - mock at server level
        with patch("app.server.get_entities", return_value=[]):
            result = await search_entities_tool("")
            assert "all entities" in result["query"]

        # Test with special characters in query - mock at server level
        test_entities = [{"entity_id": "sensor.test_temp", "state": "22", "attributes": {}}]
        with patch("app.server.get_entities", return_value=test_entities):
            result = await search_entities_tool("test_temp")
            assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_tool_parameter_combinations(self):
        """Test tools with various parameter combinations."""
        from app.server import (
            get_error_log,
            get_version,
            restart_ha,
            system_overview,
        )

        # Test get_version - mock at server level
        with patch("app.server.get_hass_version", return_value="2024.1.0"):
            result = await get_version()
            assert result == "2024.1.0"

        # Test restart_ha - mock at server level
        with patch("app.server.restart_home_assistant", return_value={"status": "ok"}):
            result = await restart_ha()
            assert "status" in result

        # Test system_overview - mock at server level
        mock_overview = {"total_entities": 10, "domains": {}}
        with patch("app.server.get_system_overview", return_value=mock_overview):
            result = await system_overview()
            assert result == mock_overview

        # Test get_error_log with different parameters - mock at server level
        mock_log = {"log_text": "test", "error_count": 0, "warning_count": 0}
        with patch("app.server.get_hass_error_log", return_value=mock_log):
            result = await get_error_log()
            assert result == mock_log

            result = await get_error_log(max_chars=1000)
            assert result == mock_log

            result = await get_error_log(tail_lines=50)
            assert result == mock_log