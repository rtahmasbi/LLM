# Install
```sh
curl -fsSL https://openclaw.ai/install.sh | bash
```

# Run onboarding

```sh
openclaw onboard --install-daemon

openclaw gateway status
```

# Open the dashboard

```sh
openclaw dashboard
```

# gateway stop
```sh
openclaw gateway stop
```



You can also run it through `Ollama`, check commands there.



```sh
openclaw channels enable --channel slack --account default
openclaw channels status

openclaw gateway restart
openclaw channels list --all

openclaw status --deep

openclaw message send \
  --channel slack \
  --target channel:<CHANNEL_ID> \
  --message "Hello from OpenClaw"
  

```