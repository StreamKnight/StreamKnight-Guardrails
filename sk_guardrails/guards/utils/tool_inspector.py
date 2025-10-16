# guards/utils/tool_inspector.py
import logging
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tool_inspector")

async def get_mcp_tools(url: str):
    """
    Connects to an MCP server and returns the list of available tools.
    """
    try:
        async with streamablehttp_client(url) as streams:
            read_stream, write_stream = streams[:2]

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()

                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools or []
                ]
    except Exception as e:
        logger.error(f"Error inspecting MCP server: {e}")
        return []
