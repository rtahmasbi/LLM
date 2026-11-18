# NVIDIA OpenShell
https://github.com/NVIDIA/OpenShell


OpenShell is the safe, private runtime for autonomous AI agents. It provides sandboxed execution environments that protect your data, credentials, and infrastructure — governed by declarative YAML policies that prevent unauthorized file access, data exfiltration, and uncontrolled network activity.

OpenShell is built agent-first. The project ships with agent skills for everything from gateway troubleshooting to policy generation, and we expect contributors to use them.



https://docs.nvidia.com/openshell/latest/home

AI agents are most useful when they can read files, install packages, call APIs, and use credentials. That same access can create material risk. OpenShell is designed for this tradeoff: preserve agent capability while enforcing explicit controls over what the agent can access.



## How OpenShell Works
https://docs.nvidia.com/openshell/latest/about/how-it-works

OpenShell is built around three stable runtime components: the CLI, the Gateway, and the Supervisor.




# Install
```
sudo usermod -aG docker $USER

curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh


systemctl --user status openshell-gateway --no-pager

```


```sh
openshell gateway start
openshell gateway list

# Kali
openshell sandbox create --from docker.io/kalilinux/kali-rolling:latest -- /bin/bash


#Use --no-keep for an ephemeral command
#Use --restart-policy on-failure to replace the runtime after a nonzero main process exit, or --restart-policy always to replace it after every exit:


# Every sandbox requires a gateway. Register or select one before running sandbox commands:
openshell gateway add http://127.0.0.1:18080 --local --name local
openshell gateway select local

# cpu and memory
openshell sandbox create --from registry.example.com/your-org/claude-agent:latest --cpu 2 --memory 4Gi -- claude


# Sandbox Images
openshell sandbox create --from nvcr.io/nvidia/base/ubuntu:24.04
openshell sandbox create --from ./rootfs.tar


docker build -t my-image:latest .
openshell sandbox create --from my-image:latest


# Connect to a Sandbox
openshell sandbox connect my-sandbox


# Execute a Command in a Sandbox
openshell sandbox exec -n my-sandbox -- ls -la /workspace

# Run an interactive shell with a TTY:
openshell sandbox exec -n my-sandbox --tty -- /bin/bash


# Set Environment Variables
openshell sandbox create --env API_KEY=sk-test --env DEBUG=1 -- my-agent

export SESSION_TOKEN="..."
openshell sandbox create --env-from SESSION_TOKEN -- my-agent
openshell sandbox create --env-from AGENT_TOKEN=SESSION_TOKEN -- my-agent


# Expose Long Running Services
openshell sandbox create \
  --name my-sandbox \
  --from docker.io/library/python:3.14-slim \
  --expose 8080 \
  --detach \
  -- python3 -m http.server 8080 --bind 127.0.0.1




openshell service list
openshell logs my-sandbox


# Port Forwarding
openshell forward start 8000 my-sandbox
openshell forward start 8000 my-sandbox -d    # run in background

openshell forward list


# Transfer Files
openshell sandbox upload my-sandbox ./src
openshell sandbox download my-sandbox output ./local


openshell sandbox stop my-sandbox
openshell sandbox start my-sandbox
openshell sandbox delete my-sandbox


# Sandbox Lifecycle





```


# python
```py
from openshell import SandboxClient

with SandboxClient.from_active_cluster() as client:
    templates = client.sandbox_templates()

    templates.create(
        workspace="default",
        name="python",
        image="registry.example.com/agents/python:latest",
    )

    sandbox = client.create_from_template(workspace="default", workload_template="python")


```
