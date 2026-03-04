# install
### 1. Server

```bash
cd server
pip install -r requirements.txt
 
# make sure your .env is filled

python -m server.main
```


### 2. Client (on the machine to diagnose)

```bash
cd client
pip install -r requirements.txt
export MCP_SERVER_URL=https://abcd-1234.ngrok-free.app

python -m client.main --issue "Load average is 18, everything is slow"
# OR
python -m client.main --server https://abcd-1234.ngrok-free.app

```

