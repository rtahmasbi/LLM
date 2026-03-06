from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client.src.tools import TOOL_REGISTRY

mcp = FastMCP("SysDiag Client")

for tool_name, fn in TOOL_REGISTRY.items():
    mcp.tool(name=tool_name)(fn)


if __name__ == "__main__":
    mcp.run(transport="stdio")
