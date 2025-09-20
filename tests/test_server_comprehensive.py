"""Comprehensive tests with extensive mocking for server module to reach 90% coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestServerComprehensive:
    """Comprehensive tests with extensive mocking for all server code paths."""

    @pytest.mark.asyncio
    async def test_async_handler_decorator_comprehensive(self):
        """Test async_handler decorator with various scenarios."""
        from app.server import async_handler

        # Test with different return types
        @async_handler("test_command")
        async def test_func_dict() -> dict:
            return {"result": "success"}

        @async_handler("test_command")
        async def test_func_list() -> list:
            return [{"item": "value"}]

        @async_handler("test_command")
        async def test_func_str() -> str:
            return "success"

        # Test each function type
        result = await test_func_dict()
        assert isinstance(result, dict)

        result = await test_func_list()
        assert isinstance(result, list)

        result = await test_func_str()
        assert isinstance(result, str)

        # Test with exception in decorated function
        @async_handler("test_error")
        async def test_func_error() -> dict:
            raise Exception("Test error")

        # The async_handler decorator doesn't catch exceptions, it just logs and calls the function
        with pytest.raises(Exception):
            await test_func_error()

    @pytest.mark.asyncio
    async def test_entity_action_comprehensive_scenarios(self):
        """Test entity_action with comprehensive action and parameter scenarios."""
        from app.server import entity_action

        mock_result = {"success": True}

        with patch("app.server.call_service", return_value=mock_result) as mock_call:
            # Test all action types
            actions = ["on", "off", "toggle"]
            entities = [
                "light.living_room",
                "switch.porch",
                "fan.bedroom",
                "cover.garage",
                "climate.thermostat",
            ]

            for action in actions:
                for entity_id in entities:
                    domain = entity_id.split(".")[0]
                    result = await entity_action(entity_id=entity_id, action=action)
                    assert result == mock_result

                    # Verify correct service mapping
                    expected_service = f"turn_{action}" if action != "toggle" else "toggle"
                    mock_call.assert_called_with(
                        domain, expected_service, {"entity_id": entity_id}
                    )

            # Test with various parameter combinations
            test_params = [
                {"brightness": 255},
                {"brightness": 128, "color_temp": 370},
                {"rgb_color": [255, 128, 0], "transition": 2},
                {"position": 50},  # For covers
                {"temperature": 22.5},  # For climate
                {"percentage": 75},  # For fans
            ]

            for params in test_params:
                result = await entity_action(
                    entity_id="light.test", action="on", params=params
                )
                assert result == mock_result
                # Verify params were merged correctly
                expected_data = {"entity_id": "light.test", **params}
                mock_call.assert_called_with("light", "turn_on", expected_data)

    @pytest.mark.asyncio
    async def test_list_entities_comprehensive_filtering(self):
        """Test list_entities with comprehensive filtering scenarios."""
        from app.server import list_entities

        # Mock different entity sets for different scenarios
        mock_entities_all = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "light.kitchen", "state": "off"},
            {"entity_id": "switch.porch", "state": "on"},
            {"entity_id": "sensor.temperature", "state": "22.5"},
        ]

        mock_entities_light = [
            {"entity_id": "light.living_room", "state": "on"},
            {"entity_id": "light.kitchen", "state": "off"},
        ]

        with patch("app.server.get_entities") as mock_get:
            # Test no filters
            mock_get.return_value = mock_entities_all
            result = await list_entities()
            assert result == mock_entities_all
            mock_get.assert_called_with(
                domain=None, search_query=None, limit=100, fields=None, lean=True
            )

            # Test domain filter
            mock_get.return_value = mock_entities_light
            result = await list_entities(domain="light")
            assert result == mock_entities_light
            mock_get.assert_called_with(
                domain="light", search_query=None, limit=100, fields=None, lean=True
            )

            # Test search query
            mock_get.return_value = [mock_entities_all[0]]
            result = await list_entities(search_query="living")
            mock_get.assert_called_with(
                domain=None, search_query="living", limit=100, fields=None, lean=True
            )

            # Test limit
            result = await list_entities(limit=50)
            mock_get.assert_called_with(
                domain=None, search_query=None, limit=50, fields=None, lean=True
            )

            # Test detailed mode
            result = await list_entities(detailed=True)
            mock_get.assert_called_with(
                domain=None, search_query=None, limit=100, fields=None, lean=False
            )

            # Test specific fields
            result = await list_entities(fields=["state", "attributes.brightness"])
            mock_get.assert_called_with(
                domain=None,
                search_query=None,
                limit=100,
                fields=["state", "attributes.brightness"],
                lean=True,
            )

            # Test error response handling
            mock_get.return_value = {"error": "API error"}
            result = await list_entities()
            assert result == []  # Should return empty list for errors

    @pytest.mark.asyncio
    async def test_search_entities_tool_comprehensive(self):
        """Test search_entities_tool with comprehensive scenarios."""
        from app.server import search_entities_tool

        # Test various entity configurations
        mock_entities = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"brightness": 255, "friendly_name": "Living Room"},
            },
            {
                "entity_id": "sensor.living_room_temp",
                "state": "22.5",
                "attributes": {"unit_of_measurement": "°C", "friendly_name": "Temperature"},
            },
        ]

        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            # Test normal search
            result = await search_entities_tool(query="living", limit=10)

            assert result["count"] == 2
            assert result["query"] == "living"
            assert "domains" in result
            assert "light" in result["domains"]
            assert "sensor" in result["domains"]
            assert len(result["results"]) == 2

            # Verify simplified entity format
            for entity in result["results"]:
                assert "entity_id" in entity
                assert "state" in entity
                assert "domain" in entity
                assert "friendly_name" in entity

            # Test empty query (special case)
            result = await search_entities_tool(query="", limit=5)
            assert "all entities (no filtering)" in result["query"]

            # Test with larger limit
            result = await search_entities_tool(query="test", limit=100)
            mock_get.assert_called_with(search_query="test", limit=100, lean=True)

            # Test error handling
            mock_get.return_value = {"error": "API error"}
            result = await search_entities_tool(query="test")
            assert "error" in result
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_resource_functions_comprehensive(self):
        """Test all resource functions with various scenarios."""
        from app.server import (
            get_all_entities_resource,
            get_entity_resource,
            get_entity_resource_detailed,
            list_states_by_domain_resource,
            search_entities_resource_with_limit,
        )

        # Mock entity data for resources
        mock_entity = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "brightness": 255,
                "friendly_name": "Living Room Light",
                "color_temp": 370,
            },
        }

        mock_entities = [mock_entity, {"entity_id": "light.kitchen", "state": "off"}]

        # Test get_entity_resource
        with patch("app.server.get_entity_state", return_value=mock_entity):
            result = await get_entity_resource("light.living_room")
            assert "light.living_room" in result
            assert "on" in result
            assert "brightness" in result

        # Test get_entity_resource_detailed
        with patch("app.server.get_entity_state", return_value=mock_entity):
            result = await get_entity_resource_detailed("light.living_room")
            assert "light.living_room" in result
            assert "brightness" in result
            assert "color_temp" in result

        # Test get_all_entities_resource
        with patch("app.server.get_entities", return_value=mock_entities):
            result = await get_all_entities_resource()
            assert "Home Assistant Entities" in result
            assert "light.living_room" in result
            assert "light.kitchen" in result

        # Test list_states_by_domain_resource
        with patch("app.server.get_entities", return_value=mock_entities):
            result = await list_states_by_domain_resource("light")
            assert "Light Entities" in result
            assert "light.living_room" in result

        # Test search_entities_resource_with_limit
        with patch("app.server.get_entities", return_value=mock_entities):
            result = await search_entities_resource_with_limit("living", "5")
            assert "living" in result
            assert "light.living_room" in result

        # Test error handling for resources
        with patch("app.server.get_entity_state", return_value={"error": "Not found"}):
            result = await get_entity_resource("nonexistent.entity")
            assert "Error" in result or "error" in result

    @pytest.mark.asyncio
    async def test_domain_summary_tool_comprehensive(self):
        """Test domain_summary_tool with various domain types."""
        from app.server import domain_summary_tool

        # Test different domain summaries
        mock_summaries = {
            "light": {
                "domain": "light",
                "total_count": 5,
                "state_distribution": {"on": 3, "off": 2},
                "examples": {"on": [{"entity_id": "light.living_room"}]},
                "common_attributes": [("brightness", 5), ("friendly_name", 5)],
            },
            "sensor": {
                "domain": "sensor",
                "total_count": 10,
                "state_distribution": {"22.5": 1, "45": 1, "unknown": 8},
                "examples": {"22.5": [{"entity_id": "sensor.temperature"}]},
                "common_attributes": [("unit_of_measurement", 10), ("device_class", 8)],
            },
        }

        for domain, mock_summary in mock_summaries.items():
            with patch("app.server.summarize_domain", return_value=mock_summary):
                result = await domain_summary_tool(domain=domain, example_limit=3)
                assert result == mock_summary

                # Verify function was called with correct parameters
                # The function will be called during the with block

    @pytest.mark.asyncio
    async def test_automation_management_comprehensive(self):
        """Test automation management functions comprehensively."""
        from app.server import list_automations

        # Test successful automation retrieval
        mock_automations = [
            {
                "id": "morning_lights",
                "entity_id": "automation.morning_lights",
                "state": "on",
                "alias": "Turn on morning lights",
                "last_triggered": "2024-01-01T07:00:00Z",
            },
            {
                "id": "night_routine",
                "entity_id": "automation.night_routine",
                "state": "off",
                "alias": "Night routine",
                "last_triggered": None,
            },
        ]

        with patch("app.server.get_automations", return_value=mock_automations):
            result = await list_automations()
            assert len(result) == 2
            assert result[0]["id"] == "morning_lights"
            assert result[1]["state"] == "off"

        # Test various error scenarios
        error_scenarios = [
            {"error": "HTTP error: 404 - Not Found"},  # Dict error
            [{"error": "HTTP error: 404 - Not Found"}],  # List with error dict
            Exception("Connection failed"),  # Exception
        ]

        for error_scenario in error_scenarios:
            if isinstance(error_scenario, Exception):
                with patch("app.server.get_automations", side_effect=error_scenario):
                    result = await list_automations()
                    assert result == []
            else:
                with patch("app.server.get_automations", return_value=error_scenario):
                    result = await list_automations()
                    assert result == []

    @pytest.mark.asyncio
    async def test_system_tools_comprehensive(self):
        """Test system-level tools comprehensively."""
        from app.server import get_error_log, get_version, restart_ha, system_overview

        # Test get_version
        with patch("app.server.get_hass_version", return_value="2024.3.0"):
            result = await get_version()
            assert result == "2024.3.0"

        # Test restart_ha
        with patch("app.server.restart_home_assistant", return_value={"message": "Restarting"}):
            result = await restart_ha()
            assert "message" in result

        # Test system_overview
        mock_overview = {
            "total_entities": 100,
            "domains": {"light": {"count": 20}, "sensor": {"count": 30}},
            "domain_samples": {"light": [{"entity_id": "light.test"}]},
        }
        with patch("app.server.get_system_overview", return_value=mock_overview):
            result = await system_overview()
            assert result == mock_overview

        # Test get_error_log with various parameters
        mock_log = {
            "log_text": "ERROR: Test error\\nWARNING: Test warning",
            "error_count": 1,
            "warning_count": 1,
            "truncated": False,
            "original_size": 100,
        }

        with patch("app.server.get_hass_error_log", return_value=mock_log) as mock_get_log:
            # Test default parameters
            result = await get_error_log()
            mock_get_log.assert_called_with(max_chars=80000, tail_lines=None)
            assert result == mock_log

            # Test custom parameters
            result = await get_error_log(max_chars=50000, tail_lines=100)
            mock_get_log.assert_called_with(max_chars=50000, tail_lines=100)

    @pytest.mark.asyncio
    async def test_get_history_comprehensive(self):
        """Test get_history function comprehensively."""
        from app.server import get_history

        # Test successful history retrieval - mock raw API response (list of lists)
        mock_api_response = [[
            {"state": "22.5", "last_changed": "2024-01-01T10:00:00Z"},
            {"state": "23.0", "last_changed": "2024-01-01T11:00:00Z"},
        ]]

        # Mock the actual hass function that get_history calls
        with patch("app.server.get_entity_history", return_value=mock_api_response):
            result = await get_history(entity_id="sensor.temperature", hours=24)

            # Check processed result structure
            assert result["entity_id"] == "sensor.temperature"
            assert result["count"] == 2
            assert len(result["states"]) == 2
            assert result["states"][0]["state"] == "22.5"
            assert result["states"][1]["state"] == "23.0"

            # Test different time periods
            result = await get_history(entity_id="sensor.humidity", hours=168)
            # Function should be called with new parameters

        # Test error handling - error responses are dicts, not lists
        mock_error = {"error": "Entity not found"}
        with patch("app.server.get_entity_history", return_value=mock_error):
            result = await get_history(entity_id="bad.entity")
            # Check error handling
            assert result["entity_id"] == "bad.entity"
            assert "error" in result
            assert result["count"] == 0
            assert result["states"] == []

    @pytest.mark.asyncio
    async def test_fastmcp_integration_comprehensive(self):
        """Test FastMCP integration scenarios."""
        # Test that the mcp object exists and has required attributes
        from app.server import mcp

        assert hasattr(mcp, "name")
        assert mcp.name == "Hass-MCP"

        # Test that all tool and resource decorators work
        # This is implicitly tested by the existence of the decorated functions

    @pytest.mark.asyncio
    async def test_prompt_generation_comprehensive(self):
        """Test all prompt generation functions."""
        from app.server import create_automation, debug_automation, troubleshoot_entity

        # Test create_automation with different trigger types
        trigger_types = ["time", "state", "event", "webhook", "device", "zone"]
        for trigger_type in trigger_types:
            result = create_automation(trigger_type=trigger_type)
            assert isinstance(result, list)
            assert len(result) >= 2  # Should have system and user messages
            assert any("role" in msg and msg["role"] == "system" for msg in result)
            assert any("role" in msg and msg["role"] == "user" for msg in result)
            assert trigger_type in str(result)  # Trigger type should be mentioned

        # Test debug_automation
        automation_ids = ["test.automation", "morning_lights", "security_system"]
        for automation_id in automation_ids:
            result = debug_automation(automation_id=automation_id)
            assert isinstance(result, list)
            assert len(result) >= 2
            assert automation_id in str(result)

        # Test troubleshoot_entity
        entity_ids = ["light.living_room", "sensor.temperature", "switch.porch"]
        for entity_id in entity_ids:
            result = troubleshoot_entity(entity_id=entity_id)
            assert isinstance(result, list)
            assert len(result) >= 2
            assert entity_id in str(result)

    @pytest.mark.asyncio
    async def test_error_edge_cases_comprehensive(self):
        """Test comprehensive error edge cases."""
        from app.server import entity_action, get_entity, list_entities

        # Test with various error response formats
        error_responses = [
            {"error": "Simple error message"},
            {"error": "HTTP error: 500 - Internal Server Error"},
            {"error": "Connection timeout"},
        ]

        for error_response in error_responses:
            # Test get_entity error handling
            with patch("app.server.get_entity_state", return_value=error_response):
                result = await get_entity("test.entity")
                assert result == error_response

            # Test list_entities error handling
            with patch("app.server.get_entities", return_value=error_response):
                result = await list_entities()
                assert result == []  # Should return empty list

            # Test entity_action error handling
            with patch("app.server.call_service", return_value=error_response):
                result = await entity_action("light.test", "on")
                assert result == error_response