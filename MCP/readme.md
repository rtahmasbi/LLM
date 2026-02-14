For the app and server, we can use:

- `app = Flask()`     -> `app.run`
- `mcp = FastMCP()`   -> `mcp.run`
- `app = FastAPI()`   -> `uvicorn.run(app, host="0.0.0.0", port=5000)`
- `app = Starlette()` -> `uvicorn.run(app, host="0.0.0.0", port=5000)`

For OpenAI, we also need to use `ngrok`


