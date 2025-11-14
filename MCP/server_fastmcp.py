
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="CalculatorServer")

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)

"""
curl -i -X POST -H "Content-Type: application/json" -d '{"key":"val"}' http://localhost:8080/appname/path
"""
