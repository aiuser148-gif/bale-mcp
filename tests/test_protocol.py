import json

import pytest
from mcp import Client

from bale_mcp.server import mcp

pytestmark = pytest.mark.anyio


async def test_wire_format_is_one_text_block_and_no_output_schema(api):
    """Real client, no conftest patch: results are one compact text block, no structuredContent, no outputSchema."""
    api["getMe"] = {"id": 1, "username": "bot", "first_name": "B", "is_bot": True}
    async with Client(mcp) as c:
        tools = (await c.list_tools()).tools
        assert tools and all(t.output_schema is None for t in tools)
        assert all("title" not in p for t in tools for p in t.input_schema.get("properties", {}).values())
        result = await c.call_tool("bot_get_me", {})
    assert result.structured_content is None
    assert len(result.content) == 1 and result.content[0].type == "text"
    assert result.content[0].text == json.dumps(
        json.loads(result.content[0].text), ensure_ascii=False, separators=(",", ":")
    )
