# Ouro — MLX-Native Model Runner

> **What Ollama is to llama.cpp, Ouro is to mlx-lm.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](https://developer.apple.com/silicon/)

Ouro is a developer-friendly CLI and API server for running large language models natively on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms). Pull models from Hugging Face, run them locally, and expose an OpenAI-compatible API — all with a single tool.

---

## Installation

```bash
pip install ouro
```

> **Requires Apple Silicon (M1/M2/M3/M4) and Python 3.11+**

---

## Quick Start

```bash
# Pull a model from Hugging Face
ouro pull mlx-community/Llama-3.2-3B-Instruct-4bit

# Run an interactive chat session
ouro run mlx-community/Llama-3.2-3B-Instruct-4bit

# Start the OpenAI-compatible API server
ouro serve mlx-community/Llama-3.2-3B-Instruct-4bit

# List locally available models
ouro list

# Check running servers
ouro ps

# Remove a model
ouro rm mlx-community/Llama-3.2-3B-Instruct-4bit
```

---

## Commands

| Command | Description |
|---------|-------------|
| `ouro pull <model>` | Download a model from Hugging Face Hub |
| `ouro list` | List all locally available models |
| `ouro rm <model>` | Remove a model from local storage |
| `ouro run <model>` | Start an interactive chat session with a model |
| `ouro serve <model>` | Start an OpenAI-compatible HTTP server |
| `ouro ps` | List all running Ouro server processes |
| `ouro stop <model>` | Stop a running model server |
| `ouro create <name>` | Create a custom model variant from a Modelfile |

### Command Details

#### `ouro pull`
```bash
ouro pull mlx-community/Llama-3.2-3B-Instruct-4bit
ouro pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

#### `ouro run`
```bash
# Interactive mode
ouro run mlx-community/Llama-3.2-3B-Instruct-4bit

# Single prompt
ouro run mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "Explain quantum computing in simple terms"
```

#### `ouro serve`
```bash
# Start server on default port 11434
ouro serve mlx-community/Llama-3.2-3B-Instruct-4bit

# Custom host/port
ouro serve mlx-community/Llama-3.2-3B-Instruct-4bit --host 0.0.0.0 --port 8080
```

#### `ouro create`
```bash
# Create a custom model from a Modelfile
ouro create my-assistant --file ./Modelfile
```

---

## OpenAI API Compatibility

Ouro exposes an OpenAI-compatible REST API when you run `ouro serve`. Drop it into any OpenAI SDK or tool by changing the base URL.

### Chat Completions

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello! What can you do?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Streaming

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Write a haiku about MLX"}],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:11434/v1/models
```

### Python SDK (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What is Apple Silicon?"}],
)
print(response.choices[0].message.content)
```

---

## Modelfile Format

Create custom model variants with system prompts, parameters, and templates using a `Modelfile`:

```dockerfile
# Modelfile
FROM mlx-community/Llama-3.2-3B-Instruct-4bit

SYSTEM """
You are a helpful coding assistant. You specialize in Python and always
provide clean, well-documented code examples.
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER max_tokens 2048

TEMPLATE """
{{ .System }}

User: {{ .Prompt }}
Assistant: {{ .Response }}
"""
```

Build your custom model:

```bash
ouro create my-coding-assistant --file ./Modelfile
ouro run my-coding-assistant
```

### Modelfile Directives

| Directive | Description |
|-----------|-------------|
| `FROM` | Base model to use (Hugging Face model ID) |
| `SYSTEM` | System prompt injected at the start of every conversation |
| `PARAMETER` | Set generation parameters (temperature, top_p, max_tokens, etc.) |
| `TEMPLATE` | Custom prompt template format |

---

## Configuration

Ouro reads configuration from `~/.ouro/config.yaml`:

```yaml
# ~/.ouro/config.yaml

# Directory where models are stored
models_dir: ~/.ouro/models

# Default server settings
server:
  host: 127.0.0.1
  port: 11434

# Default generation parameters
defaults:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  repetition_penalty: 1.1

# Hugging Face settings
huggingface:
  # Optional: HF token for gated models
  token: null
  # Cache directory for HF downloads
  cache_dir: ~/.cache/huggingface
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OURO_MODELS_DIR` | Override model storage directory |
| `OURO_HOST` | Override default server host |
| `OURO_PORT` | Override default server port |
| `HF_TOKEN` | Hugging Face API token for gated models |

---

## Requirements

- **Hardware**: Apple Silicon (M1, M2, M3, or M4 chip)
- **OS**: macOS 13.3 (Ventura) or later
- **Python**: 3.11 or later
- **MLX**: Installed automatically as a dependency

---

## Why Ouro?

| Feature | Ouro | Ollama |
|---------|------|--------|
| Backend | mlx-lm (Apple-native) | llama.cpp |
| Hardware | Apple Silicon only | Cross-platform |
| Memory | Unified memory (Metal GPU) | CPU + optional GPU |
| Speed | Native MLX performance | GGUF quantized |
| Modelfile | ✅ | ✅ |
| OpenAI API | ✅ | ✅ |
| HF Hub integration | ✅ | ✅ |

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/Cyber521k/Ouro).

---

## License

Apache-2.0 © Nous Research

See [LICENSE](LICENSE) for details.
