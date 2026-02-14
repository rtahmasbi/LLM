"""
MCP Server for OpenAI Platform (HTTP/SSE)

Installation:
pip install fastapi uvicorn pydantic

Usage:
python server.py

For OpenAI Platform, you need a public URL:
1. Run: python server_for_openai.py
2. Use ngrok: ngrok http 5000
3. Add ngrok URL to OpenAI: https://xxxx.ngrok.io
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import Optional

app = FastAPI()

# Enable CORS for OpenAI Platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

# MCP Protocol Implementation
@app.get("/")
async def root():
    """Root endpoint - MCP server info."""
    return {
        "jsonrpc": "2.0",
        "serverInfo": {
            "name": "multiply-server",
            "version": "1.0.0"
        },
        "capabilities": {
            "tools": {}
        }
    }

@app.post("/")
async def handle_mcp_request(request: Request):
    """Handle MCP JSON-RPC requests."""
    try:
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "multiply-server",
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "multiply",
                            "description": "Multiply two integers and return the result",
                            "inputSchema": {
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
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "multiply":
                a = arguments.get("a")
                b = arguments.get("b")
                
                if not isinstance(a, int) or not isinstance(b, int):
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Both arguments must be integers"
                        }
                    }
                
                result = multiply(a, b)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "server": "multiply-mcp-server"}

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("MCP Server for OpenAI Platform")
    print("=" * 60)
    print("Local URL: http://0.0.0.0:5000")
    print("\nFor OpenAI Platform, create a public URL with ngrok:")
    print("  1. Install ngrok: https://ngrok.com/download")
    print("  2. Run: ngrok http 5000")
    print("  3. Copy the https URL (e.g., https://abc123.ngrok.io)")
    print("  4. Add that URL to OpenAI Platform")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)


"""
python server_for_openai.py
OR (after making a little change)
uvicorn main:app --host 0.0.0.0 --port 5000

"""