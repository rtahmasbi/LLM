import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

url = "http://localhost:8000/sse"


async def main():
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:

            # initialize MCP connection
            await session.initialize()

            # list tools
            tools = await session.list_tools()
            print("TOOLS:", tools)

            # call tool
            result = await session.call_tool(
                "multiply",
                {"a": 6, "b": 7}
            )

            print("RESULT:", result)


asyncio.run(main())

"""
python client_fastmcp.py

"""