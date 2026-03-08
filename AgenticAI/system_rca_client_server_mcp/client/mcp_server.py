from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client.src.tools import TOOL_REGISTRY

mcp = FastMCP("SysDiag Client")

# this loop is similar to:
# @mcp.tool(name="list_processes")
# def list_processes(...): ...

for tool_name, fn in TOOL_REGISTRY.items():
    mcp.tool(name=tool_name)(fn)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8001"))
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
