# AGENTS.md

**FastAPI** — Modern, fast web framework for building APIs with Python type hints.

## §1 Environment & Commands

**Install (editable with all extras):**
```bash
pip install -e '.[standard,standard-no-fastapi-cloud-cli,all]'
```

**Run tests:**
```bash
pytest
pytest tests/test_tutorial  # Tutorial examples
pytest -v -s  # Verbose with prints
```

**Run type checking:**
```bash
mypy fastapi
```

**Run linting:**
```bash
ruff check .
ruff format .
```

**Build docs:**
```bash
python scripts/docs.py live
```

**Translation management:**
```bash
python scripts/translate.py
```

## §2 Architecture Map

**Core request handling flow:**
```
FastAPI (applications.py)
  ↓
APIRouter (routing.py)
  ↓
solve_dependencies (dependencies/utils.py)
  ↓
Pydantic validation (_compat/v2.py)
  ↓
OpenAPI schema (openapi/utils.py)
```

**Key modules:**
- `fastapi/applications.py` — FastAPI class, app initialization
- `fastapi/routing.py` — APIRouter, route definitions, endpoint handling
- `fastapi/dependencies/utils.py` — Dependency injection solver (HIGH COUPLING)
- `fastapi/_compat/v2.py` — Pydantic v1/v2 compatibility layer (HIGH COUPLING)
- `fastapi/openapi/utils.py` — OpenAPI schema generation
- `fastapi/params.py` — Query, Path, Body, Header parameter definitions
- `fastapi/responses.py` — Response classes (JSONResponse, StreamingResponse, etc.)
- `fastapi/background.py` — Background task handling
- `fastapi/exceptions.py` — HTTPException, validation error handlers
- `fastapi/security/` — OAuth2, API key, HTTP auth utilities
- `fastapi/middleware/` — CORS, trusted host middleware

**Documentation structure:**
- `docs/` — Main documentation (Markdown)
- `docs_src/` — Tutorial code examples (versioned by Python version)
- `scripts/docs.py` — Documentation build tooling
- `scripts/translate.py` — i18n translation management

## §3 Code Conventions

**Path parameters and dependency injection:**
```python
# ✓ Use Annotated for parameters
@app.get("/items/{item_id}")
async def read_item(item_id: Annotated[int, Path(gt=0)]):
    ...

# ✓ Use Depends for dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def read_users(db: Annotated[Session, Depends(get_db)]):
    ...

# ✗ Don't use bare type hints for FastAPI parameters
@app.get("/items/{item_id}")
async def read_item(item_id: int):  # Missing Path()
    ...
```

**Response models and status codes:**
```python
# ✓ Use response_model for output validation
@app.post("/items/", response_model=Item, status_code=201)
async def create_item(item: Item):
    ...

# ✓ Use responses parameter for additional responses
@app.get(
    "/items/{item_id}",
    responses={404: {"model": Message}},
)
async def read_item(item_id: int):
    ...

# ✗ Don't return raw dicts when response_model is set
@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    return {"name": item.name}  # Should return Item instance
```

**Async/sync handling:**
```python
# ✓ Use async for I/O-bound operations
@app.get("/items/")
async def read_items():
    items = await fetch_items_from_db()
    return items

# ✓ Use sync for CPU-bound or blocking operations
@app.get("/compute/")
def compute_result():
    return heavy_computation()

# ✗ Don't mix sync calls in async functions
@app.get("/bad/")
async def bad_endpoint():
    time.sleep(1)  # Blocks event loop
    return {"status": "done"}
```

**Pydantic model inheritance:**
```python
# ✓ Use proper Pydantic v2 BaseModel
from pydantic import BaseModel, Field

class ItemBase(BaseModel):
    name: str
    description: str | None = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

# ✗ Don't use deprecated Pydantic v1 patterns
class Item(BaseModel):
    class Config:  # Use model_config in v2
        orm_mode = True
```

**Testing patterns:**
```python
# ✓ Use TestClient for endpoint testing
from fastapi.testclient import TestClient

def test_read_item():
    client = TestClient(app)
    response = client.get("/items/1")
    assert response.status_code == 200

# ✓ Test with dependency overrides
def test_with_override():
    def override_dependency():
        return "test"
    
    app.dependency_overrides[get_dependency] = override_dependency
    client = TestClient(app)
    # ... test
    app.dependency_overrides.clear()

# ✗ Don't make real HTTP requests in tests
def test_bad():
    import requests
    response = requests.get("http://localhost:8000/items/1")
```

## §4 PR & Contribution Rules

**Branch pattern:**
```
feat/<description>  # New features
fix/<description>   # Bug fixes
docs/<description>  # Documentation
refactor/<description>
```

**Merge strategy:** Squash commits on merge.

**Commit messages:**
- No strict conventional commits enforced, but descriptive messages preferred
- Include issue number when applicable: "Fix dependency resolution #1234"

**PR checklist:**
1. Add tests for new features/fixes in `tests/` or `tests/test_tutorial/`
2. Update docs in `docs/` if user-facing changes
3. Add tutorial examples in `docs_src/` with Python version suffixes (_py310.py, _py39.py)
4. Run `mypy fastapi` — must pass without errors
5. Run `ruff check .` and `ruff format .` — must be clean
6. Run full test suite — must pass
7. Update `CHANGELOG.md` if significant change

**Tutorial code examples:**
- Create versioned files: `tutorial001_py310.py`, `tutorial001_py39.py`, etc.
- Keep examples minimal and focused on one concept
- Each example must be runnable standalone

**Documentation translation:**
- Use `scripts/translate.py` to manage translations
- Don't manually edit translated files unless fixing translation script

## §5 Anti-Patterns & Gotchas

**❌ Modifying `dependencies/utils.py` without extensive testing:**
- This module is tightly coupled to `routing.py` and `_compat/v2.py` (12 and 11 co-changes)
- Changes affect dependency resolution across entire framework
- MUST test with nested dependencies, sync/async mixing, and background tasks

**❌ Breaking Pydantic v1/v2 compatibility in `_compat/v2.py`:**
- FastAPI supports both Pydantic v1 and v2
- Changes to `_compat/v2.py` affect `dependencies/utils.py` and `routing.py` (11 and 10 co-changes)
- Test with both `pydantic>=2.0.0` and `pydantic<2.0.0` environments

**❌ Incorrect OpenAPI schema generation:**
- `openapi/utils.py` is coupled to `dependencies/utils.py` and `routing.py` (11 and 10 co-changes)
- Schema generation depends on parameter extraction logic
- Verify generated OpenAPI JSON matches expected structure

**❌ Using sync I/O operations in async routes:**
```python
# ✗ WRONG — blocks event loop
@app.get("/bad/")
async def bad_endpoint():
    with open("file.txt") as f:  # Blocking I/O
        content = f.read()
    return content

# ✓ CORRECT — use async I/O
@app.get("/good/")
async def good_endpoint():
    async with aiofiles.open("file.txt") as f:
        content = await f.read()
    return content
```

**❌ Modifying route registration without updating OpenAPI schema:**
- Route decorators in `routing.py` must properly register with schema generator
- Test that new route types appear correctly in `/docs` and `/redoc`

**❌ Breaking background task lifecycle:**
```python
# ✗ WRONG — task runs after response closed
@app.post("/send-notification/")
async def send_notification(background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, "user@example.com")
    # Task might access closed resources
    return {"message": "Notification sent"}

# ✓ CORRECT — ensure resources available for background task
@app.post("/send-notification/")
async def send_notification(
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)]
):
    # Copy data needed by task, don't pass db session directly
    user_email = db.query(User).first().email
    background_tasks.add_task(send_email, user_email)
    return {"message": "Notification sent"}
```

**❌ Incorrect dependency override cleanup in tests:**
```python
# ✗ WRONG — leaks overrides between tests
def test_one():
    app.dependency_overrides[get_db] = override_get_db
    # No cleanup

def test_two():
    # Still has override from test_one!
    ...

# ✓ CORRECT — always clean up
def test_one():
    app.dependency_overrides[get_db] = override_get_db
    try:
        # Test code
        ...
    finally:
        app.dependency_overrides.clear()
```

**❌ Adding middleware without considering order:**
```python
# ✗ WRONG — CORS middleware after routes won't work
app.include_router(router)
app.add_middleware(CORSMiddleware, ...)  # Too late

# ✓ CORRECT — middleware before routes
app.add_middleware(CORSMiddleware, ...)
app.include_router(router)
```

**⚠️ Low type annotation coverage areas (be cautious):**
- `docs_src/additional_responses/*.py`
- `docs_src/additional_status_codes/*.py`
- `docs_src/advanced_middleware/*.py`

These are tutorial examples, not production code. Don't replicate their annotation patterns in core.

## §6 File Change Risk Map

**🔴 High-churn files (proceed with extreme caution):**
- `fastapi/__init__.py` — Public API surface, imports, version
- `fastapi/dependencies/utils.py` — Dependency solver (couples to routing, compat, openapi)
- `fastapi/routing.py` — Route handling (couples to dependencies, compat, openapi)
- `fastapi/_compat/v2.py` — Pydantic v1/v2 bridge (couples to dependencies, routing)
- `fastapi/openapi/utils.py` — Schema generation (couples to dependencies, routing)
- `fastapi/applications.py` — FastAPI app class
- `scripts/translate.py` — Translation tooling
- `scripts/docs.py` — Documentation build

**Coupling hotspots (change together):**
1. `dependencies/utils.py` ↔ `routing.py` (12 co-changes)
2. `_compat/v2.py` ↔ `dependencies/utils.py` (11 co-changes)
3. `dependencies/utils.py` ↔ `openapi/utils.py` (11 co-changes)
4. `_compat/v2.py` ↔ `routing.py` (10 co-changes)
5. `openapi/utils.py` ↔ `routing.py` (10 co-changes)

**Change strategy for coupled files:**
- If modifying `dependencies/utils.py`, review/test `routing.py`, `_compat/v2.py`, `openapi/utils.py`
- If modifying `routing.py`, review `dependencies/utils.py`, `openapi/utils.py`, `_compat/v2.py`
- Run full test suite, especially `tests/test_tutorial/test_dependency_injection/`

**🟢 Stable files (safer to modify):**
- `fastapi/cli.py` — CLI entry point
- `fastapi/responses.py` — Response classes
- `docs_src/server_sent_events/*.py` — SSE tutorial examples

## §7 Testing Conventions

**Test organization:**
```
tests/
├── test_*.py              # Core framework tests
├── test_tutorial/         # Tutorial example tests
│   ├── test_*.py
│   └── test_*_py310.py   # Python version-specific
└── conftest.py           # Shared fixtures
```

**Running tests:**
```bash
# Full suite
pytest

# Specific test file
pytest tests/test_dependency_injection.py

# Specific test
pytest tests/test_dependency_injection.py::test_dependency_gets_exception

# Tutorial examples
pytest tests/test_tutorial/

# With coverage
pytest --cov=fastapi --cov-report=html
```

**Test patterns:**
```python
# ✓ Use TestClient for integration tests
from fastapi.testclient import TestClient

def test_read_main():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

# ✓ Test dependency overrides
def test_override_dependency():
    def override():
        return {"db": "mock"}
    
    app.dependency_overrides[get_db] = override
    client = TestClient(app)
    response = client.get("/items/")
    assert response.status_code == 200
    app.dependency_overrides.clear()

# ✓ Test validation errors
def test_validation_error():
    client = TestClient(app)
    response = client.post("/items/", json={"name": 123})  # Invalid type
    assert response.status_code == 422
    assert "validation" in response.json()["detail"][0]["type"]

# ✓ Test background tasks
def test_background_tasks():
    executed = []
    
    def task(name: str):
        executed.append(name)
    
    @app.get("/send")
    def send(background_tasks: BackgroundTasks):
        background_tasks.add_task(task, "test")
        return {"message": "sent"}
    
    client = TestClient(app)
    response = client.get("/send")
    assert response.status_code == 200
    assert "test" in executed
```

**Testing async endpoints:**
```python
# ✓ TestClient handles async automatically
def test_async_endpoint():
    @app.get("/async")
    async def async_route():
        return {"async": True}
    
    client = TestClient(app)
    response = client.get("/async")
    assert response.status_code == 200

# ✓ Use pytest-asyncio for async test functions if needed
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

**Testing OpenAPI schema:**
```python
def test_openapi_schema():
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/items/" in schema["paths"]
    assert schema["paths"]["/items/"]["get"]["responses"]["200"]
```

**Testing WebSocket endpoints:**
```python
def test_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        data = websocket.receive_json(improve_agent_programmatically)
```
