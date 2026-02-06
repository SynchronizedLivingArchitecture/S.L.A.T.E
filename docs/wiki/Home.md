# S.L.A.T.E. Wiki

Welcome to the **S.L.A.T.E.** (Synchronized Living Architecture for Transformation and Evolution) documentation wiki.

## What is SLATE?

SLATE is a local-first AI agent orchestration framework that:

- Coordinates multiple AI models (Ollama, Foundry Local, external APIs)
- Runs entirely on your machine - no cloud dependencies
- Automatically optimizes for your hardware (GPU/CPU)
- Manages complex multi-step workflows

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Getting Started](Getting-Started) | Installation and first steps |
| [Architecture](Architecture) | System design and components |
| [Agent System](Agents) | ALPHA, BETA, GAMMA, DELTA agents |
| [AI Backends](AI-Backends) | Ollama, Foundry Local, external APIs |
| [CLI Reference](CLI-Reference) | Command-line tools and options |
| [Configuration](Configuration) | Settings and customization |
| [Development](Development) | Contributing and extending SLATE |
| [Troubleshooting](Troubleshooting) | Common issues and solutions |

## Key Concepts

### Local-First
All SLATE services bind to `127.0.0.1`. Your code, prompts, and data never leave your machine.

### Multi-Agent Architecture
SLATE uses specialized agents for different tasks:
- **ALPHA**: Code generation and implementation
- **BETA**: Testing and validation
- **GAMMA**: Planning and research
- **DELTA**: External integrations

### Hardware Optimization
SLATE auto-detects your hardware and configures:
- PyTorch settings (TF32, BF16, Flash Attention)
- GPU memory allocation
- Model selection based on VRAM

## Getting Help

- [GitHub Issues](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/issues) - Bug reports and feature requests
- [Discussions](https://github.com/SynchronizedLivingArchitecture/S.L.A.T.E/discussions) - Questions and community support

## Version

Current version: **2.4.0**
