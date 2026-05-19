# GAIA × Hermes — Self-Evaluating Coding Assistant Setup

Use Hermes as an autonomous coding agent to build websites from hand-drawn wireframes using a local coding model, while GAIA continuously evaluates the output and provides feedback through a stronger critic model.

This setup uses:

- **Hermes** → coding agent
- **GAIA** → refinement and evaluation loop
- **Lemonade Server** → local model hosting
- **Qwen3.5-9B-GGUF** → coder model
- **Qwen3.6-35B-GGUF** → critic model

My setup:

- **Windows** → Lemonade Server + GAIA
- **WSL** → Hermes

---

# Prerequisites

1. Lemonade Server  - https://lemonade-server.ai/
2. GAIA Agent  - https://github.com/amd/gaia
3. Hermes Agent  - https://hermes-agent.nousresearch.com/

---

# 1. Install Lemonade Server

Install Lemonade Server and download the required models based on your hardware capacity.

Recommended models:

- `Qwen3.5-9B-GGUF` → coding
- `Qwen3.6-35B-GGUF` → evaluation / critic

Check that Lemonade is running:

```bash
lemonade status
```

By default, Lemonade usually runs on port:

```text
13305
```

Change the maximum loaded models:

```text
lemonade config set max_loaded_models=2
```

---

# 2. Install GAIA

Clone and build GAIA on Windows:

```bash
git clone https://github.com/meghsat/gaia.git
cd gaia
git checkout hermes-critic
pip install -e .
```

---

# 3. Install Hermes in WSL

Clone and build Hermes inside WSL:

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

Create the virtual environment:

```bash
uv venv venv --python 3.11
```

Activate it:

```bash
export VIRTUAL_ENV="$(pwd)/venv"
```

Install dependencies:

```bash
uv pip install -e ".[all,dev]"
```

Optional:

```bash
npm install
```

---

# 4. Expose Lemonade Server to WSL

## Check Lemonade Port

```bash
lemonade status
```

Default port:

```text
13305
```

---

## Create Windows Port Proxy

Run in **PowerShell (Administrator)**:

```powershell
netsh interface portproxy add v4tov4 `
listenaddress=0.0.0.0 `
listenport=13305 `
connectaddress=127.0.0.1 `
connectport=13305
```

---

## Enable IP Helper Service

```powershell
Set-Service iphlpsvc -StartupType Automatic
Start-Service iphlpsvc
```

---

## Verify Port Binding

```powershell
netstat -ano | findstr 13305
```

You should see:

```text
0.0.0.0:13305
```

---

## Find Windows Host IP from WSL

Inside WSL:

```bash
ip route | grep default
```

Example output:

```text
default via 172.17.128.1 dev eth0
```

In this example:

```text
172.17.128.1
```

is the Windows host IP.

---

## Test Connection from WSL

```bash
curl http://172.17.128.1:13305/api/v1
```

It should return Lemonade Server details.

---

# 5. Start the GAIA Critic

Open a PowerShell terminal on Windows:

```bash
gaia refine serve --critic-model Qwen3.6-35B-A3B-GGUF
```

---

# 6. Configure Hermes

Open:

```text
/home/<USER>/.hermes/config.yaml
```

Append the Lemonade provider configuration:

```yaml
custom_providers:
  - name: lemonade_server
    base_url: http://172.17.128.1:13305/api/v1
    model: Qwen3.6-35B-A3B-GGUF

    models:
      Qwen3.5-35B-A3B-GGUF:
        context_length: 190000

      Qwen3.5-9B-GGUF:
        context_length: 260000
```

Replace `172.17.128.1` with your actual Windows host IP if different.

---

# 7. Start Hermes

Inside WSL:

```bash
hermes
```

To switch models:

```text
/model
```

Then select the desired model from the dropdown.

---

# 8. Using the GAIA Feedback Loop

Provide Hermes with:

- hand-drawn wireframes
- website assets
- prompts
- coding instructions
- custom skills

Workflow:

1. Hermes generates the implementation using the coder model.
2. GAIA evaluates the output using the critic model.
3. Feedback is returned to Hermes.
4. Hermes refines the implementation.
5. The loop continues until:
   - the result is satisfactory, or
   - the iteration limit is reached.

This creates a fully local self-evaluating coding workflow powered by Hermes + GAIA.
