"""
MCP Server with SSE for OpenAI Platform

Installation:
pip install mcp starlette uvicorn sse-starlette

Usage:
python server.py

Then add to OpenAI Platform:
URL: http://YOUR_IP:5000/sse
(or use ngrok/cloudflare tunnel for public URL)
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
from sse_starlette.sse import EventSourceResponse
import uvicorn

# Create MCP server instance
mcp_server = Server("multiply-server")

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="multiply",
            description="Multiply two integers and return the result",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "First integer to multiply"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second integer to multiply"
                    }
                },
                "required": ["a", "b"]
            }
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "multiply":
        a = arguments.get("a")
        b = arguments.get("b")
        
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("Both arguments must be integers")
        
        result = multiply(a, b)
        return [TextContent(
            type="text",
            text=str(result)
        )]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def sse_handler(request):
    """Handle SSE connection for MCP protocol."""
    async def event_generator():
        # Send initial connection event
        yield {
            "event": "message",
            "data": json.dumps({"type": "connection", "status": "connected"})
        }
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
            yield {
                "event": "ping",
                "data": json.dumps({"type": "ping"})
            }
    
    return EventSourceResponse(event_generator())

async def health_check(request):
    """Health check endpoint."""
    return Response(
        content=json.dumps({"status": "healthy", "server": "multiply-mcp-server"}),
        media_type="application/json"
    )

# Create Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route('/sse', sse_handler),
        Route('/health', health_check),
    ]
)

if __name__ == "__main__":
    print("=" * 50)
    print("MCP Server with SSE starting...")
    print("Server URL: http://0.0.0.0:5000/sse")
    print("=" * 50)
    print("\nAdd this URL to OpenAI Platform:")
    print("http://YOUR_IP_ADDRESS:5000/sse")
    print("\nFor public access, use ngrok:")
    print("ngrok http 5000")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)

#

"""
pip install mcp starlette uvicorn sse-starlette



Add to OpenAI Platform:
If running locally:

URL: http://localhost:5000/sse



python server.py


"""
