    from mcp.server.fastmcp import FastMCP
    import os

    mcp = FastMCP("FileSystemServer")

    @mcp.tool()
    def write_file(path: str, content: str):
        """Creates a new file or overwrites an existing one with specified content."""
        with open(path, 'w') as f:
            f.write(content)
        return f"File '{path}' written successfully."


    if __name__ == "__main__":
        mcp.run()



"""
in ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": [
        "--project-dir",
        "C:\\path\\to\\your\\specific\\project",
        "--reference-project",
        "docs=C:\\path\\to\\documentation",
        "--reference-project",
        "examples=C:\\path\\to\\examples",
        "--log-level",
        "INFO"
      ]
    }
  }
}

"""