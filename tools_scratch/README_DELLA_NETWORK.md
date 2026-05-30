# Internet Access on Della Compute Nodes

Della compute nodes cannot directly access the internet. The cluster proxy blocks HTTPS tunneling (403 Forbidden). This guide sets up a reverse SSH tunnel through your local machine.

## Prerequisites

- Local machine with internet access
- SSH access to Della
- `tinyproxy` or similar proxy on local machine

## Setup Steps

### 1. Install proxy on your local machine

**Mac:**
```bash
brew install tinyproxy
```

**Linux:**
```bash
sudo apt install tinyproxy
```

### 2. Start tinyproxy on local machine

```bash
# Create config (use a port not in use, e.g., 18888)
echo "Port 18888
Listen 127.0.0.1
Timeout 600
Allow 127.0.0.1" > /tmp/tinyproxy.conf

tinyproxy -d -c /tmp/tinyproxy.conf
```

Verify it works locally:
```bash
curl -x http://localhost:18888 -s -o /dev/null -w "%{http_code}" https://www.google.com
# Should return 200
```

### 3. Create SSH tunnel from local to Della login node

In a new terminal on your local machine:
```bash
ssh -R 18888:localhost:18888 hw0171@della.princeton.edu
```

Note which login node you land on (e.g., `della9`):
```bash
hostname  # e.g., della9.princeton.edu
```

### 4. On compute node: forward to login node

On your interactive compute node session (e.g., `della-l08g2`):
```bash
ssh -f -N -L 18888:localhost:18888 della9  # Replace della9 with your login node
```

### 5. Set environment variables on compute node

```bash
export HTTP_PROXY=http://localhost:18888
export HTTPS_PROXY=http://localhost:18888
export http_proxy=http://localhost:18888
export https_proxy=http://localhost:18888

# IMPORTANT: Exclude localhost from proxy (for local services like vLLM)
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
```

### 6. Test connectivity

```bash
# Test external access
curl -s -o /dev/null -w "%{http_code}" https://www.google.com
# Should return 200

# Test DuckDuckGo search (for tool calling)
python -c "from ddgs import DDGS; print(list(DDGS(proxy='http://localhost:18888').text('test', max_results=1)))"
```

## Quick Reference

| Component | Location | Port |
|-----------|----------|------|
| tinyproxy | Local machine | 18888 |
| SSH tunnel | Local -> Della login | 18888 |
| Port forward | Compute -> Login | 18888 |

## Troubleshooting

### "000" HTTP code
- Proxy not running or tunnel broken
- Check tinyproxy is running on local machine
- Check SSH tunnel is still connected

### vLLM returns 404
- Forgot to set `no_proxy=localhost,127.0.0.1`
- Localhost requests are going through proxy instead of direct

### Search returns empty results
- Use new `ddgs` package, not old `duckduckgo_search`:
  ```bash
  pip install ddgs
  ```

### Port already in use
- Choose a different port (e.g., 19999)
- Update all commands accordingly
