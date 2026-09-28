# In-App AI Agent — Pure Java LLM (GPULlama3.java + TornadoVM + LangChain4j)

A Spring Boot proof of concept that runs an LLM **fully locally and offline** inside the JVM — no Python,
no Ollama, no external API. GGUF models are loaded directly by **GPULlama3.java**, matrix-vector
multiplication is offloaded to the GPU by **TornadoVM**, and orchestration (chat memory, prompt templates,
streaming) is handled by **LangChain4j**.

```
Browser / client
      |  REST + SSE (/api/chat, /api/chat/stream, /api/analyze)
      v
Spring Boot 3.5 (JDK 25)
  AiController
      |
  LangChain4j AiServices ── ChatAgent (per-session memory)   TextAnalyst (stateless)
      |
  BlockingChatModel ──> GatedStreamingChatModel   (1 inference thread + bounded queue)
                              |
                   GPULlama3StreamingChatModel  (langchain4j-gpu-llama3)
                              |
                   GPULlama3.java (inference engine, GGUF file)
                         /                 \
              TornadoVM (OpenCL/PTX)     Pure Java CPU (Vector API)
                  GPU                          CPU
```

## Stack & versions

| Component | Version | Notes |
|---|---|---|
| JDK | OpenJDK 25.0.2 | `langchain4j-gpu-llama3` is compiled for JDK 25 (class major 69) |
| Spring Boot | 3.5.16 | web + validation |
| LangChain4j | 1.20.1 | `AiServices`, `MessageWindowChatMemory` |
| langchain4j-gpu-llama3 | 1.20.1-beta30 | pulls in `gpu-llama3` 0.4.0-jdk25 |
| TornadoVM SDK | 4.0.1-jdk25 (OpenCL) | must match the TornadoVM API bundled in `gpu-llama3` |
| Default model | Llama 3.2 1B Instruct FP16 (2.5 GB) | from `beehive-lab` on Hugging Face |

## Quick start (Windows / PowerShell)

1. **JDK 25** — extract OpenJDK 25 from <https://jdk.java.net/archive/> to `%USERPROFILE%\.jdks\jdk-25.0.2`
   (or set `JAVA25_HOME`). This is deliberately separate from `JAVA_HOME`.
2. **TornadoVM SDK** (GPU mode only) — download
   `tornadovm-4.0.1-jdk25-opencl-windows-amd64.zip` from
   <https://github.com/beehive-lab/TornadoVM/releases/tag/v4.0.1-jdk25> and extract it to
   `.tools\tornadovm-4.0.1-jdk25-opencl` (or set `TORNADOVM_HOME`). Use the `-cuda-` build for NVIDIA (PTX).
3. **Model**:
   ```powershell
   .\scripts\download-model.ps1                   # Llama 3.2 1B FP16 (default)
   .\scripts\download-model.ps1 deepseek-r1-1.5b  # DeepSeek-R1-Distill-Qwen-1.5B Q8_0
   .\scripts\download-model.ps1 phi3-mini         # Phi-3 mini 4k Q4
   .\scripts\download-model.ps1 llama-3b          # Llama 3.2 3B FP16
   ```
   For a non-default model, set `$env:LLM_MODEL_PATH = '...\models\<file>.gguf'`.
4. **Build & run**:
   ```powershell
   $env:JAVA_HOME = "$HOME\.jdks\jdk-25.0.2"; mvn -DskipTests package
   .\run.ps1            # CPU (default)
   .\run.ps1 gpu        # TornadoVM / GPU
   .\run.ps1 devices    # list OpenCL devices detected by TornadoVM
   ```
5. Open <http://localhost:8080> — streaming chat UI plus a text analysis panel.

> `run.ps1` builds its own JVM argfile from `tornado-argfile.template` because the SDK's bundled `tornado`
> launcher breaks on paths containing spaces. The app runs from the `target/classes;target/lib/*` classpath
> (not a Spring Boot fat jar) so the TornadoVM modules on `--module-path` don't clash with the nested-jar
> classloader.

## API

| Method | Path | Body | Description |
|---|---|---|---|
| GET | `/api/info` | – | model, backend, number of queued requests |
| POST | `/api/chat` | `{"sessionId?":"..","message":".."}` | full reply + `elapsedMs` |
| POST | `/api/chat/stream` | same | SSE: `session`, `queued`, `token`*, `done` / `error` |
| DELETE | `/api/chat/{sessionId}` | – | clear session memory |
| POST | `/api/analyze` | `{"task":"SUMMARIZE","text":".."}` | tasks: `SUMMARIZE`, `SENTIMENT`, `KEYWORDS`, `ENTITIES`, `TRANSLATE_ID`, `TRANSLATE_EN` |

```bash
curl -s localhost:8080/api/chat -H 'Content-Type: application/json' \
  -d '{"sessionId":"s1","message":"Hi, my name is Budi"}'
curl -N localhost:8080/api/chat/stream -H 'Content-Type: application/json' \
  -d '{"sessionId":"s1","message":"What is my name?"}'
```

## Benchmark (dev laptop, Llama 3.2 1B FP16, single request)

Hardware: AMD Ryzen APU with **Radeon gfx90c (integrated GPU sharing system RAM)**.

| Mode | Time to first token | Generation speed | Total per answer |
|---|---|---|---|
| CPU (pure Java, Vector API) | 8.7 s | ~42 tok/s | **~14 s** |
| GPU TornadoVM OpenCL (warm) | 50 s | ~9.5 tok/s | ~93 s |
| GPU TornadoVM OpenCL (first request, incl. kernel JIT) | 119 s | ~4.6 tok/s | ~175 s |

**Takeaway:** on an integrated GPU that shares memory bandwidth with the CPU, TornadoVM is about 6x
*slower*, so `run.ps1` defaults to CPU. GPU mode is meant for discrete GPUs (NVIDIA via PTX/OpenCL, discrete
AMD Radeon); the LangChain4j docs report ~49 tok/s on an NVIDIA GPU. Model load time: ~3 s (CPU) vs ~35 s
(GPU, including TornadoVM plan initialization).

## Design decisions

- **One inference thread + bounded queue** (`GatedStreamingChatModel`). GPULlama3 keeps its KV cache and
  TornadoVM execution plan inside the model instance, so it is not thread-safe. Every generation (and the model
  load) runs on the `llm-inference` thread; when the queue is full (`llm.queue-capacity`) the API returns **503**.
- **A single copy of the model weights.** `BlockingChatModel` is a synchronous view over the same streaming
  model, so the sync and streaming endpoints don't load the model onto the GPU twice.
- **The UI never looks dead:** the `queued` SSE event reports the queue position, the UI shows a timer until
  the first token, and it detects connections that drop before `done`.

## Configuration (`application.yml` / env)

| Property | Env | Default |
|---|---|---|
| `llm.model-path` | `LLM_MODEL_PATH` | `models/Llama-3.2-1B-Instruct-FP16.gguf` |
| `llm.on-gpu` | `LLM_ON_GPU` (set by `run.ps1`) | `true` |
| `llm.max-tokens` | – | 1024 (KV cache context length) |
| `llm.memory-messages` | – | 8 messages per session |
| `llm.queue-capacity` | – | 8 |
| `llm.request-timeout` | – | 5m |
| `server.port` | `PORT` | 8080 |

## POC limitations

- GPULlama3 does not support **tool calling** yet, so the "agent" here is a system prompt plus conversation memory.
- Chat memory is in-memory only (lost on restart).
- The 1B model is fast but answer quality is limited (it sometimes hallucinates); use 3B or
  DeepSeek-R1-Distill if your hardware allows.
- `langchain4j-gpu-llama3` is still a beta release; `gpu-llama3` is a shaded jar that also bundles log4j/Graal.

## Tests

```powershell
$env:JAVA_HOME = "$HOME\.jdks\jdk-25.0.2"; mvn test
```
`GatedStreamingChatModelTest` verifies that generations never overlap and that a full queue is rejected with
`InferenceBusyException` (no GPU or model required).
