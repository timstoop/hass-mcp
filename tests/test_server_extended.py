"""Extended tests for the server module to increase coverage."""

from unittest.mock import AsyncMock, patch

import pytest


class TestServerExtended:
    """Extended tests for MCP server functionality."""

    @pytest.mark.asyncio
    async def test_get_version_tool(self):
        """Test the get_version tool."""
        from app.server import get_version

        with patch("app.server.get_hass_version", return_value="2024.3.0") as mock_version:
            version = await get_version()
            assert version == "2024.3.0"
            mock_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_action_turn_on(self):
        """Test entity action turn_on."""
        from app.server import entity_action

        mock_result = {"success": True}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            result = await entity_action(entity_id="light.living_room", action="on")

            assert result == mock_result
            mock_call.assert_called_once_with("light", "turn_on", {"entity_id": "light.living_room"})

    @pytest.mark.asyncio
    async def test_entity_action_turn_off(self):
        """Test entity action turn_off."""
        from app.server import entity_action

        mock_result = {"success": True}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            result = await entity_action(entity_id="switch.kitchen", action="off")

            assert result == mock_result
            mock_call.assert_called_once_with("switch", "turn_off", {"entity_id": "switch.kitchen"})

    @pytest.mark.asyncio
    async def test_entity_action_toggle(self):
        """Test entity action toggle."""
        from app.server import entity_action

        mock_result = {"success": True}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            result = await entity_action(entity_id="light.bedroom", action="toggle")

            assert result == mock_result
            mock_call.assert_called_once_with("light", "toggle", {"entity_id": "light.bedroom"})

    @pytest.mark.asyncio
    async def test_entity_action_with_params(self):
        """Test entity action with additional parameters."""
        from app.server import entity_action

        mock_result = {"success": True}
        params = {"brightness": 255, "color_name": "red"}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            result = await entity_action(
                entity_id="light.living_room",
                action="on",
                params=params
            )

            assert result == mock_result
            expected_data = {"entity_id": "light.living_room", **params}
            mock_call.assert_called_once_with("light", "turn_on", expected_data)

    @pytest.mark.asyncio
    async def test_call_service_tool(self):
        """Test the call_service_tool function."""
        from app.server import call_service_tool

        mock_result = {"success": True}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            result = await call_service_tool(
                domain="light",
                service="turn_on",
                data={"entity_id": "light.living_room", "brightness": 255}
            )

            assert result == mock_result
            mock_call.assert_called_once_with(
                "light",
                "turn_on",
                {"entity_id": "light.living_room", "brightness": 255}
            )

    @pytest.mark.asyncio
    async def test_restart_ha_tool(self):
        """Test the restart_ha tool."""
        from app.server import restart_ha

        mock_result = {"message": "Restarting Home Assistant"}

        with patch("app.server.restart_home_assistant", return_value=mock_result) as mock_restart:
            result = await restart_ha()

            assert result == mock_result
            mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_overview_tool(self):
        """Test the system_overview tool."""
        from app.server import system_overview

        mock_overview = {
            "total_entities": 50,
            "domains": {"light": 10, "switch": 5},
            "domain_samples": {}
        }

        with patch("app.server.get_system_overview", return_value=mock_overview) as mock_get:
            result = await system_overview()

            assert result == mock_overview
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history_tool(self):
        """Test the get_history tool."""
        from app.server import get_history

        # The get_entity_history function returns a list of lists from the API
        mock_api_response = [[{"state": "22.5", "last_changed": "2024-01-01T12:00:00Z"}]]

        with patch("app.server.get_entity_history", return_value=mock_api_response) as mock_get:
            result = await get_history(entity_id="sensor.temperature", hours=24)

            # Check the processed result structure
            assert result["entity_id"] == "sensor.temperature"
            assert result["count"] == 1
            assert len(result["states"]) == 1
            assert result["states"][0]["state"] == "22.5"
            mock_get.assert_called_once_with("sensor.temperature", 24)

    @pytest.mark.asyncio
    async def test_list_entities_with_domain(self):
        """Test list_entities with domain filter."""
        from app.server import list_entities

        mock_entities = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "light.kitchen", "state": "off"}
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            result = await list_entities(domain="light")

            assert result == mock_entities
            mock_get.assert_called_once_with(
                domain="light",
                search_query=None,
                limit=100,
                fields=None,
                lean=True
            )

    @pytest.mark.asyncio
    async def test_list_entities_with_search(self):
        """Test list_entities with search query."""
        from app.server import list_entities

        mock_entities = [
            {"entity_id": "light.living_room", "state": "on"}
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            result = await list_entities(search_query="living", limit=50)

            assert result == mock_entities
            mock_get.assert_called_once_with(
                domain=None,
                search_query="living",
                limit=50,
                fields=None,
                lean=True
            )

    @pytest.mark.asyncio
    async def test_entity_resource_functions(self):
        """Test entity resource functions."""
        from app.server import get_entity_resource, get_entity_resource_detailed

        mock_entity = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"brightness": 255}
        }

        with patch("app.server.get_entity_state", return_value=mock_entity) as mock_get:
            # Test basic resource
            result = await get_entity_resource("light.living_room")
            assert "light.living_room" in result
            assert "on" in result

            # Test detailed resource
            result_detailed = await get_entity_resource_detailed("light.living_room")
            assert "light.living_room" in result_detailed
            assert "brightness" in result_detailed

    @pytest.mark.asyncio
    async def test_all_entities_resource(self):
        """Test get_all_entities_resource function."""
        from app.server import get_all_entities_resource

        mock_entities = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "switch.kitchen", "state": "off"}
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            result = await get_all_entities_resource()

            assert "Home Assistant Entities" in result
            assert "light.living_room" in result
            assert "switch.kitchen" in result

    @pytest.mark.asyncio
    async def test_domain_resource_functions(self):
        """Test domain-specific resource functions."""
        from app.server import list_states_by_domain_resource

        mock_entities = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "light.kitchen", "state": "off", "attributes": {}}
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            result = await list_states_by_domain_resource("light")

            assert "Light Entities" in result
            assert "light.living_room" in result
            assert "light.kitchen" in result

    @pytest.mark.asyncio
    async def test_search_resource_with_limit(self):
        """Test search resource with limit."""
        from app.server import search_entities_resource_with_limit

        mock_entities = [
            {"entity_id": "light.living_room", "state": "on"}
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            result = await search_entities_resource_with_limit("living", "10")

            assert "living" in result
            assert "light.living_room" in result

    @pytest.mark.asyncio
    async def test_prompt_functions(self):
        """Test prompt functions exist and return lists of messages."""
        from app.server import create_automation, debug_automation, troubleshoot_entity

        # Test create_automation prompt with required trigger_type
        prompt = create_automation(trigger_type="time")  # No await needed - returns list
        assert isinstance(prompt, list)
        assert len(prompt) > 0

        # Test debug_automation prompt with required automation_id
        prompt = debug_automation(automation_id="test_automation")  # No await needed
        assert isinstance(prompt, list)
        assert len(prompt) > 0

        # Test troubleshoot_entity prompt with required entity_id
        prompt = troubleshoot_entity(entity_id="light.living_room")  # No await needed
        assert isinstance(prompt, list)
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_tools(self):
        """Test error handling in various tools."""
        from app.server import get_entity, list_entities

        # Test error response handling
        mock_error_response = {"error": "Entity not found"}

        with patch("app.server.get_entity_state", return_value=mock_error_response):
            result = await get_entity("nonexistent.entity")
            assert result == mock_error_response

        with patch("app.server.get_entities", return_value=mock_error_response):
            result = await list_entities()
            # Should return empty list for error cases
            assert result == []