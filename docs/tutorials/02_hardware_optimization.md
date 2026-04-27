# Tutorial: Hardware-Aware RAG Optimization

Advanced RAG Studio is unique because it understands the hardware it runs on. This guide shows you how to tune the system for maximum performance on **Apple Silicon (M3/M4)** or **NVIDIA GPUs**.

## 💻 1. Inspecting your Profile
Go to the **Hardware & Models** view. The system has automatically tiered your machine.
- **Apple Silicon High (M3 Max/Ultra)**: You can run 8B-14B models with high context windows.
- **NVIDIA High (RTX 4090/A6000)**: You can prioritize raw throughput and use larger embedding models.

## 🚀 2. Model Selection Strategy
Your hardware tier determines which models are best for the "HyDE" and "Generation" stages.

| Task | Recommendation | Why? |
|---|---|---|
| **Embeddings** | `nomic-embed-text` | 8k context window, optimized for Apple MLX/Metal. |
| **HyDE (Fast)** | `llama3.2:1b` | Instant generation (~100ms) on Apple Silicon, perfect for low-latency search. |
| **Generation** | `llama3.1:8b` | The current gold standard for local reasoning. |

## ⚙️ 3. Tuning the Pipeline Config
Go to the **Pipeline Config** view to apply these optimizations:

### For Apple Silicon (Unified Memory)
1. **RRF K**: Keep at `60` (default). Unified memory handles the rank merging extremely fast.
2. **Top-K**: You can safely increase this to `10` or `15` without a significant latency hit.
3. **HyDE Temperature**: Set to `0.3` to keep the hypothetical document grounded in technical patterns.

## 🔋 4. Monitoring Performance
When you run a query, check the **Pipeline Trace** latency badges:
- **Routing**: Should be `<20ms`.
- **HyDE**: Should be `<500ms` on Apple Silicon.
- **Search**: Should be `<50ms` for indices under 10k chunks.

If HyDE is slow, use a smaller model (like `gemma2:2b`) in the Config view to speed up the transformation stage.
