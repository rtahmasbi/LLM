# AGENTS.md

## §1 Environment & Commands
- Install the project dependencies and setup the development environment:
  ```bash
  pip install -e '.[standard,standard-no-fastapi-cloud-cli,all]'
  ```

## §2 Architecture Map
- **Test Suite:**
  - Includes files like `tests/test_security_openid_connect_optional.py` and `tests/test_dependency_yield_scope.py`.

- **Nested Models and Testing:**
  - Includes files like `docs_src/body_nested_models/tutorial004_py310.py` and `tests/test_generate_unique_id_function.py`.

- **API Endpoint Handling:**
  - Includes files like `docs_src/metadata/tutorial003_py310.py` and `docs_src/custom_response/tutorial004_py310.py`.

- **Test Client Setup:**
  - Includes files like `tests/test_security_scopes_sub_dependency.py` and `tests/test_tutorial/test_query_params_str_validations/test_tutorial010.py`.

- **Dependency Injection Testing:**
  - Includes files like `tests/test_dependency_partial.py` and `tests/test_dependency_class.py`.

- **State Management Generators:**
  - Includes the file `tests/test_dependency_contextmanager.py`.

- **WebSocket Handling:**
  - Includes files like `docs_src/websockets_/tutorial001_py310.py` and `tests/test_ws_dependencies.py`.

- **Dependency Management Testing:**
  - Includes files like `tests/test_dependency_yield_scope_websockets.py` and `tests/test_dependency_after_yield_streaming.py`.

- **Response Class Tests:**
  - Includes files like `tests/test_default_response_class.py` and `tests/test_custom_route_class.py`.

- **Hello World Examples:**
  - Includes files like `docs_src/first_steps/tutorial001_py310.py` and `docs_src/app_testing/app_a_py310/main.py`.

- **File Handling Functions:**
  - Includes files like `docs_src/request_files/tutorial001_03_an_py310.py` and `tests/test_optional_file_list.py`.

- **Request Handling Definitions:**
  - Includes files like `docs_src/pydantic_v1_in_v2/tutorial003_an_py310.py` and `docs_src/response_model/tutorial001_01_py310.py`.

## §3 Code Conventions
- ✅ All test functions should start with `test_`.
  ```python
  def test_read_main():
  ```

- ✅ Use `json` parameter in POST requests.
  ```python
  response = client.post('/items/', json={'id': 'foobar'})
  ```

- ✅ Assert status code and response JSON body for endpoint behavior.
  ```python
  assert response.status_code == 200
  assert response.json() == {'msg': 'Hello World'}
  ```

- ✅ Consistently use `BaseModel` from Pydantic for data modeling.
  ```python
  class Item(BaseModel): id: str, value: str
  ```

- ❌ Incorrect naming or missing setup for TestClient.
- ✅ Use fixtures for setting up TestClient instances.
  ```python
  def client_fixture(app: FastAPI):
      return TestClient(app)
  ```

## §4 PR & Contribution Rules
- **Branch Pattern:** Use `feat/<description>`.
- **Merge Strategy:** Squash merges are preferred.
- **Commit Message Structure:** Conventional commits not enforced, but clarity is key.

## §5 Anti-Patterns & Gotchas
- ❌ Improper use of optional fields in data models with Python 3.10 union types ⚠️.
- ✅ Correct use:
  ```python
  class Item(BaseModel): id: str, title: str, description: str | None = None
  ```

- ❌ Missing `await websocket.accept()` in WebSocket handlers.
- ✅ Correct implementation:
  ```python
  async def websocket(websocket: WebSocket):
      await websocket.accept()
  ```

## §6 File Change Risk Map
- **High Churn Files:**
  - `fastapi/__init__.py`
  - `fastapi/dependencies/utils.py`
  - `fastapi/routing.py`

- **Stable Files:**
  - `fastapi/cli.py`
  - `docs_src/server_sent_events/__init__.py`

- **Files with High Coupling:**
  - `fastapi/dependencies/utils.py` & `fastapi/routing.py` (12 changes together)

- **Low Type Annotation Coverage:**
  - Caution when editing `docs_src/additional_responses/__init__.py` and similar files.

## §7 Testing Conventions
- Use test clients from `TestClient` for simulating HTTP calls.
- Include both valid and invalid request scenarios in tests.
- Mock inputs where possible to test isolated logic.
- Verify all HTTP response aspects, such as headers and status codes.
