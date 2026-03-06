
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Server",
    #host="0.0.0.0",   # optional
    port=8000         # default is usually 8000
)


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="sse")
    # transport="stdio" # for LangGraph / OpenAI Agents / Claude Desktop


"""
# terminal 0
python server_fastmcp.py


to test it, you can run:
python client_fastmcp.py


# OR

# terminal 1
curl -N http://localhost:8000/sse
keep it open and copy the sessionID



# terminal 2
first initialize, then call the tool
curl -X POST "http://localhost:8000/messages/?session_id=04d9670b442a4365a15a29fd24a9322c" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {
        "name": "curl",
        "version": "1.0"
      }
    }
  }'

curl -X POST http://localhost:8000/messages/?session_id=04d9670b442a4365a15a29fd24a9322c \
  -H "Content-Type: application/json" \
  -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "tools/call",
        "params": {
          "name": "multiply",
          "arguments": {
            "a": 6,
            "b": 7
          }
        }
      }'


curl -X POST "http://localhost:8000/messages/?session_id=04d9670b442a4365a15a29fd24a9322c" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
  }'

"""
