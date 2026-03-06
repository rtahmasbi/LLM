# SysDiag MCP refactor


## Install

```bash
conda create -n sysdiag python=3.11 -y
conda activate sysdiag

pip install -r requirements.txt

pip install --force-reinstall --no-deps typing_inspection
conda run -n sysdiag pip install --force-reinstall -r requirements.txt

```

Set your OpenAI key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Run

Start the API server:

```bash
python -m server.main
```

Start a diagnosis:

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"issue": "High load average and slow response times"}'
```

Then poll status:

```bash
curl http://localhost:8000/diagnose/<session_id>
```
session_id can be found in the server log


To see all the sessions and reports:
```sh
curl http://localhost:8000/sessions | jq
```

## Notes

- The orchestrator launches `python -m client.mcp_server` as a subprocess.
