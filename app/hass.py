import functools
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

import httpx

from app.config import HA_TOKEN, HA_URL, get_ha_headers

# Set up logging
logger = logging.getLogger(__name__)

# Define a generic type for our API function return values
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


# HTTP client
class ClientManager:
    """Manages a single HTTP client instance for the application."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Get a persistent httpx client for Home Assistant API calls"""
        if self._client is None:
            logger.debug("Creating new HTTP client")
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def cleanup_client(self) -> None:
        """Close the HTTP client when shutting down"""
        if self._client:
            logger.debug("Closing HTTP client")
            await self._client.aclose()
            self._client = None


# Global client manager instance
_client_manager = ClientManager()

# Default field sets for different verbosity levels
# Lean fields for standard requests (optimized for token efficiency)
DEFAULT_LEAN_FIELDS = ["entity_id", "state", "attr.friendly_name"]

# Common fields that are typically needed for entity operations
DEFAULT_STANDARD_FIELDS = ["entity_id", "state", "attributes", "last_updated"]

# Domain-specific important attributes to include in lean responses
DOMAIN_IMPORTANT_ATTRIBUTES = {
    "light": ["brightness", "color_temp", "rgb_color", "supported_color_modes"],
    "switch": ["device_class"],
    "binary_sensor": ["device_class"],
    "sensor": ["device_class", "unit_of_measurement", "state_class"],
    "climate": ["hvac_mode", "current_temperature", "temperature", "hvac_action"],
    "media_player": ["media_title", "media_artist", "source", "volume_level"],
    "cover": ["current_position", "current_tilt_position"],
    "fan": ["percentage", "preset_mode"],
    "camera": ["entity_picture"],
    "automation": ["last_triggered"],
    "scene": [],
    "script": ["last_triggered"],
}


def _get_error_formatter(return_type: Any) -> Callable[[str], Any]:
    """Get the appropriate error formatter based on return type."""
    is_dict_return = "Dict" in str(return_type) or "dict" in str(return_type)
    is_list_return = "List" in str(return_type) or "list" in str(return_type)

    def format_error(msg: str) -> Any:
        if is_dict_return:
            return {"error": msg}
        elif is_list_return:
            return [{"error": msg}]
        else:
            return msg

    return format_error


def _handle_http_exceptions(format_error: Callable[[str], Any]) -> Callable[[Exception], Any]:
    """Handle HTTP exceptions and return formatted errors."""

    def handle_exception(e: Exception) -> Any:
        if isinstance(e, httpx.ConnectError):
            return format_error(f"Connection error: Cannot connect to Home Assistant at {HA_URL}")
        elif isinstance(e, httpx.TimeoutException):
            return format_error(
                f"Timeout error: Home Assistant at {HA_URL} did not respond in time"
            )
        elif isinstance(e, httpx.HTTPStatusError):
            return format_error(
                f"HTTP error: {e.response.status_code} - {e.response.reason_phrase}"
            )
        elif isinstance(e, httpx.RequestError):
            return format_error(f"Error connecting to Home Assistant: {e!s}")
        else:
            return format_error(f"Unexpected error: {e!s}")

    return handle_exception


def handle_api_errors(func: F) -> F:
    """
    Decorator to handle common error cases for Home Assistant API calls

    Args:
        func: The async function to decorate

    Returns:
        Wrapped function that handles errors
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Determine return type from function annotation
        return_type = inspect.signature(func).return_annotation
        format_error = _get_error_formatter(return_type)
        handle_exception = _handle_http_exceptions(format_error)

        # Check if token is available
        if not HA_TOKEN:
            return format_error(
                "No Home Assistant token provided. Please set HA_TOKEN in .env file."
            )

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return handle_exception(e)

    return cast("F", wrapper)


async def get_client() -> httpx.AsyncClient:
    """Get a persistent httpx client for Home Assistant API calls"""
    return await _client_manager.get_client()


async def cleanup_client() -> None:
    """Close the HTTP client when shutting down"""
    await _client_manager.cleanup_client()


# Direct entity retrieval function
async def get_all_entity_states() -> dict[str, dict[str, Any]]:
    """Fetch all entity states from Home Assistant"""
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
    response.raise_for_status()
    entities = cast("list[dict[str, Any]]", response.json())

    # Create a mapping for easier access
    return {entity["entity_id"]: entity for entity in entities}


def filter_fields(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """
    Filter entity data to only include requested fields

    This function helps reduce token usage by returning only requested fields.

    Args:
        data: The complete entity data dictionary
        fields: List of fields to include in the result
               - "state": Include the entity state
               - "attributes": Include all attributes
               - "attr.X": Include only attribute X (e.g. "attr.brightness")
               - "context": Include context data
               - "last_updated"/"last_changed": Include timestamp fields

    Returns:
        A filtered dictionary with only the requested fields
    """
    if not fields:
        return data

    result = {"entity_id": data["entity_id"]}

    for field in fields:
        if field == "state":
            result["state"] = data.get("state")
        elif field == "attributes":
            result["attributes"] = data.get("attributes", {})
        elif field.startswith("attr.") and len(field) > 5:
            attr_name = field[5:]
            attributes = data.get("attributes", {})
            if attr_name in attributes:
                if "attributes" not in result:
                    result["attributes"] = {}
                result["attributes"][attr_name] = attributes[attr_name]
        elif field == "context":
            if "context" in data:
                result["context"] = data["context"]
        elif field in ["last_updated", "last_changed"] and field in data:
            result[field] = data[field]

    return result


# API Functions
@handle_api_errors
async def get_hass_version() -> str:
    """Get the Home Assistant version from the API"""
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/config", headers=get_ha_headers())
    response.raise_for_status()
    data = response.json()
    return cast("str", data.get("version", "unknown"))


@handle_api_errors
async def get_entity_state(
    entity_id: str, fields: list[str] | None = None, lean: bool = False
) -> dict[str, Any]:
    """
    Get the state of a Home Assistant entity

    Args:
        entity_id: The entity ID to get
        fields: Optional list of specific fields to include in the response
        lean: If True, returns a token-efficient version with minimal fields
              (overridden by fields parameter if provided)

    Returns:
        Entity state dictionary, optionally filtered to include only specified fields
    """
    # Fetch directly
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states/{entity_id}", headers=get_ha_headers())
    response.raise_for_status()
    entity_data = cast("dict[str, Any]", response.json())

    # Apply field filtering if requested
    if fields:
        # User-specified fields take precedence
        return filter_fields(entity_data, fields)
    elif lean:
        # Build domain-specific lean fields
        lean_fields = DEFAULT_LEAN_FIELDS.copy()

        # Add domain-specific important attributes
        domain = entity_id.split(".")[0]
        if domain in DOMAIN_IMPORTANT_ATTRIBUTES:
            for attr in DOMAIN_IMPORTANT_ATTRIBUTES[domain]:
                lean_fields.append(f"attr.{attr}")

        return filter_fields(entity_data, lean_fields)
    else:
        # Return full entity data
        return entity_data


def _filter_entities_by_search(
    entities: list[dict[str, Any]], search_query: str
) -> list[dict[str, Any]]:
    """Filter entities by search query."""
    search_term = search_query.lower().strip()
    filtered_entities = []

    for entity in entities:
        # Search in entity_id
        if search_term in entity["entity_id"].lower():
            filtered_entities.append(entity)
            continue

        # Search in friendly_name
        friendly_name = entity.get("attributes", {}).get("friendly_name", "").lower()
        if friendly_name and search_term in friendly_name:
            filtered_entities.append(entity)
            continue

        # Search in other common attributes (state, area_id, etc.)
        if search_term in entity.get("state", "").lower():
            filtered_entities.append(entity)
            continue

        # Search in other attributes
        for _attr_name, attr_value in entity.get("attributes", {}).items():
            # Check if attribute value can be converted to string
            if (
                isinstance(attr_value, str | int | float | bool)
                and search_term in str(attr_value).lower()
            ):
                filtered_entities.append(entity)
                break

    return filtered_entities


def _apply_field_filtering(
    entities: list[dict[str, Any]], fields: list[str] | None, lean: bool
) -> list[dict[str, Any]]:
    """Apply field filtering to entities."""
    if fields:
        # Use explicit field list when provided
        return [filter_fields(entity, fields) for entity in entities]
    elif lean:
        # Apply domain-specific lean fields to each entity
        result = []
        for entity in entities:
            # Get the entity's domain
            entity_domain = entity["entity_id"].split(".")[0]

            # Start with basic lean fields
            lean_fields = DEFAULT_LEAN_FIELDS.copy()

            # Add domain-specific important attributes
            if entity_domain in DOMAIN_IMPORTANT_ATTRIBUTES:
                for attr in DOMAIN_IMPORTANT_ATTRIBUTES[entity_domain]:
                    lean_fields.append(f"attr.{attr}")

            # Filter and add to result
            result.append(filter_fields(entity, lean_fields))

        return result
    else:
        # Return full entities
        return entities


@handle_api_errors
async def get_entities(
    domain: str | None = None,
    search_query: str | None = None,
    limit: int = 100,
    fields: list[str] | None = None,
    lean: bool = True,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Get a list of all entities from Home Assistant with optional filtering and search

    Args:
        domain: Optional domain to filter entities by (e.g., 'light', 'switch')
        search_query: Optional case-insensitive search term to filter by entity_id, friendly_name or other attributes
        limit: Maximum number of entities to return (default: 100)
        fields: Optional list of specific fields to include in each entity
        lean: If True (default), returns token-efficient versions with minimal fields

    Returns:
        List of entity dictionaries, optionally filtered by domain and search terms,
        and optionally limited to specific fields
    """
    # Get all entities directly
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
    response.raise_for_status()
    entities = cast("list[dict[str, Any]]", response.json())

    # Filter by domain if specified
    if domain:
        entities = [entity for entity in entities if entity["entity_id"].startswith(f"{domain}.")]

    # Search if query is provided
    if search_query and search_query.strip():
        entities = _filter_entities_by_search(entities, search_query)

    # Apply the limit
    if limit > 0 and len(entities) > limit:
        entities = entities[:limit]

    # Apply field filtering
    return _apply_field_filtering(entities, fields, lean)


@handle_api_errors
async def call_service(
    domain: str, service: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Call a Home Assistant service"""
    if data is None:
        data = {}

    client = await get_client()
    response = await client.post(
        f"{HA_URL}/api/services/{domain}/{service}", headers=get_ha_headers(), json=data
    )
    response.raise_for_status()

    # Note: Cache invalidation would go here if we had entity caching

    json_response = response.json()

    # Handle cases where Home Assistant returns a list instead of dict
    # This can happen with services like automation.trigger
    if isinstance(json_response, list):
        return {"result": json_response}

    return cast("dict[str, Any]", json_response)


@handle_api_errors
async def summarize_domain(domain: str, example_limit: int = 3) -> dict[str, Any]:
    """
    Generate a summary of entities in a domain

    Args:
        domain: The domain to summarize (e.g., 'light', 'switch')
        example_limit: Maximum number of examples to include for each state

    Returns:
        Dictionary with summary information
    """
    entities = await get_entities(domain=domain)

    # Check if we got an error response
    if isinstance(entities, dict) and "error" in entities:
        return entities  # Just pass through the error

    # Type narrowing: at this point entities must be a list
    if not isinstance(entities, list):
        return {"error": "Unexpected response format from get_entities"}

    try:
        # Initialize summary data
        total_count = len(entities)
        state_counts: dict[str, int] = {}
        state_examples: dict[str, list[dict[str, str]]] = {}
        attributes_summary: dict[str, int] = {}

        # Process entities to build the summary
        for entity in entities:
            state = entity.get("state", "unknown")

            # Count states
            if state not in state_counts:
                state_counts[state] = 0
                state_examples[state] = []
            state_counts[state] += 1

            # Add examples (up to the limit)
            if len(state_examples[state]) < example_limit:
                example = {
                    "entity_id": entity["entity_id"],
                    "friendly_name": entity.get("attributes", {}).get(
                        "friendly_name", entity["entity_id"]
                    ),
                }
                state_examples[state].append(example)

            # Collect attribute keys for summary
            for attr_key in entity.get("attributes", {}):
                if attr_key not in attributes_summary:
                    attributes_summary[attr_key] = 0
                attributes_summary[attr_key] += 1

        # Create the summary
        summary = {
            "domain": domain,
            "total_count": total_count,
            "state_distribution": state_counts,
            "examples": state_examples,
            "common_attributes": sorted(
                attributes_summary.items(), key=lambda x: x[1], reverse=True
            )[:10],  # Top 10 most common attributes
        }

        return summary
    except Exception as e:
        return {"error": f"Error generating domain summary: {e!s}"}


@handle_api_errors
async def get_automations() -> list[dict[str, Any]] | dict[str, Any]:
    """Get a list of all automations from Home Assistant"""
    # Reuse the get_entities function with domain filtering
    automation_entities = await get_entities(domain="automation")

    # Check if we got an error response
    if isinstance(automation_entities, dict) and "error" in automation_entities:
        return automation_entities  # Just pass through the error

    # Type narrowing: at this point automation_entities must be a list
    if not isinstance(automation_entities, list):
        return {"error": "Unexpected response format from get_entities"}

    # Process automation entities
    result = []
    try:
        for entity in automation_entities:
            # Extract relevant information
            automation_info = {
                "id": entity["entity_id"].split(".")[1],
                "entity_id": entity["entity_id"],
                "state": entity["state"],
                "alias": entity["attributes"].get("friendly_name", entity["entity_id"]),
            }

            # Add any additional attributes that might be useful
            if "last_triggered" in entity["attributes"]:
                automation_info["last_triggered"] = entity["attributes"]["last_triggered"]

            result.append(automation_info)
    except (TypeError, KeyError) as e:
        # Handle errors in processing the entities
        return {"error": f"Error processing automation entities: {e!s}"}

    return result


@handle_api_errors
async def reload_automations() -> dict[str, Any]:
    """Reload all automations in Home Assistant"""
    return await call_service("automation", "reload", {})


@handle_api_errors
async def restart_home_assistant() -> dict[str, Any]:
    """Restart Home Assistant"""
    return await call_service("homeassistant", "restart", {})


@handle_api_errors
async def get_hass_error_log(
    max_chars: int = 80000, tail_lines: int | None = None
) -> dict[str, Any]:
    """
    Get the Home Assistant error log for troubleshooting with size limits

    Args:
        max_chars: Maximum number of characters to return (default: 80000, ~20k tokens)
        tail_lines: If specified, return only the last N lines instead of truncating by chars

    Returns:
        A dictionary containing:
        - log_text: The error log text (truncated if necessary)
        - error_count: Number of ERROR entries found (in full log)
        - warning_count: Number of WARNING entries found (in full log)
        - integration_mentions: Map of integration names to mention counts (in full log)
        - truncated: Boolean indicating if log was truncated
        - original_size: Size of the full log in characters
        - error: Error message if retrieval failed
    """
    try:
        # Call the Home Assistant API error_log endpoint
        url = f"{HA_URL}/api/error_log"
        headers = get_ha_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                full_log_text = response.text
                original_size = len(full_log_text)

                # Always analyze the full log for counts and mentions
                error_count = full_log_text.count("ERROR")
                warning_count = full_log_text.count("WARNING")

                # Extract integration mentions from full log

                integration_mentions = {}

                # Look for patterns like [mqtt], [zwave], etc.
                for match in re.finditer(r"\[([a-zA-Z0-9_]+)\]", full_log_text):
                    integration = match.group(1).lower()
                    if integration not in integration_mentions:
                        integration_mentions[integration] = 0
                    integration_mentions[integration] += 1

                # Determine the log text to return
                if tail_lines is not None:
                    # Return last N lines
                    lines = full_log_text.split("\n")
                    if len(lines) > tail_lines:
                        log_text = "\n".join(lines[-tail_lines:])
                        truncated = True
                    else:
                        log_text = full_log_text
                        truncated = False
                elif original_size > max_chars:
                    # Truncate by character count, but try to keep recent entries
                    # Take from the end to get most recent logs
                    log_text = full_log_text[-max_chars:]
                    # Try to start at a line boundary to avoid partial lines
                    newline_pos = log_text.find("\n")
                    if newline_pos > 0 and newline_pos < 1000:  # Only if reasonably close to start
                        log_text = log_text[newline_pos + 1 :]
                    truncated = True
                    # Add truncation notice
                    log_text = f"[LOG TRUNCATED - Showing last {len(log_text)} chars of {original_size} total chars]\n\n{log_text}"
                else:
                    log_text = full_log_text
                    truncated = False

                return {
                    "log_text": log_text,
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "integration_mentions": integration_mentions,
                    "truncated": truncated,
                    "original_size": original_size,
                }
            else:
                return {
                    "error": f"Error retrieving error log: {response.status_code} {response.reason_phrase}",
                    "details": response.text,
                    "log_text": "",
                    "error_count": 0,
                    "warning_count": 0,
                    "integration_mentions": {},
                    "truncated": False,
                    "original_size": 0,
                }
    except Exception as e:
        logger.error(f"Error retrieving Home Assistant error log: {e!s}")
        return {
            "error": f"Error retrieving error log: {e!s}",
            "log_text": "",
            "error_count": 0,
            "warning_count": 0,
            "integration_mentions": {},
            "truncated": False,
            "original_size": 0,
        }


@handle_api_errors
async def get_entity_history(entity_id: str, hours: int) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Get the history of an entity's state changes from Home Assistant.

    Args:
        entity_id: The entity ID to get history for.
        hours: Number of hours of history to retrieve.

    Returns:
        A list of state change objects, or an error dictionary.
    """
    client = await get_client()

    # Calculate the end time for the history lookup
    end_time = datetime.now(UTC)
    end_time_iso = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Calculate the start time for the history lookup based on end_time
    start_time = end_time - timedelta(hours=hours)
    start_time_iso = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Construct the API URL
    url = f"{HA_URL}/api/history/period/{start_time_iso}"

    # Set query parameters
    params = {
        "filter_entity_id": entity_id,
        "minimal_response": "true",
        "end_time": end_time_iso,
    }

    # Make the API call
    response = await client.get(url, headers=get_ha_headers(), params=params)
    response.raise_for_status()

    # Return the JSON response
    return cast("list[dict[str, Any]]", response.json())


def _group_entities_by_domain(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group entities by their domain."""
    domain_entities: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        domain = entity["entity_id"].split(".")[0]
        if domain not in domain_entities:
            domain_entities[domain] = []
        domain_entities[domain].append(entity)
    return domain_entities


def _create_domain_statistics(domain_entities: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Create domain statistics including state distributions."""
    domains = {}
    for domain, entities in domain_entities.items():
        # Count states for this domain
        state_counts: dict[str, int] = {}
        for entity in entities:
            state = entity.get("state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1

        domains[domain] = {
            "count": len(entities),
            "state_distribution": state_counts,
        }
    return domains


def _create_domain_samples(
    domain_entities: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Create sample entities for each domain (2-3 per domain)."""
    domain_samples = {}
    for domain, entities in domain_entities.items():
        # Take up to 3 sample entities
        domain_samples[domain] = entities[:3]
    return domain_samples


def _analyze_domain_attributes(
    domain_entities: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    """Analyze common attributes for each domain."""
    domain_attributes = {}
    for domain, entities in domain_entities.items():
        # Count attribute frequency
        attr_counts: dict[str, int] = {}
        for entity in entities:
            attributes = entity.get("attributes", {})
            for attr_name in attributes:
                attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

        # Sort by frequency and take top 10
        top_attrs = sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        domain_attributes[domain] = top_attrs

    return domain_attributes


@handle_api_errors
async def get_system_overview() -> dict[str, Any]:
    """
    Get a comprehensive overview of the entire Home Assistant system

    Returns:
        A dictionary containing:
        - total_entities: Total count of all entities
        - domains: Dictionary of domains with their entity counts and state distributions
        - domain_samples: Representative sample entities for each domain (2-3 per domain)
        - domain_attributes: Common attributes for each domain
        - area_distribution: Entities grouped by area (if available)
    """
    # Get ALL entities with minimal fields for efficiency
    client = await get_client()
    response = await client.get(f"{HA_URL}/api/states", headers=get_ha_headers())
    response.raise_for_status()
    all_entities_raw = cast("list[dict[str, Any]]", response.json())

    # Apply lean formatting to reduce token usage in the response
    all_entities = _apply_field_filtering(all_entities_raw, None, True)

    # Group entities by domain
    domain_entities = _group_entities_by_domain(all_entities)

    # Build overview using helper functions
    return {
        "total_entities": len(all_entities),
        "domains": _create_domain_statistics(domain_entities),
        "domain_samples": _create_domain_samples(domain_entities),
        "domain_attributes": _analyze_domain_attributes(domain_entities),
        "area_distribution": {},  # Could be implemented later if needed
    }
