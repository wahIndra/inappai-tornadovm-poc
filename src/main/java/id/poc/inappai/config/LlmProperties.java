package id.poc.inappai.config;

import java.nio.file.Path;
import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @param modelPath      GGUF file (FP16, Q8_0 or Q4_0; quantized weights are expanded to FP16 on load)
 * @param onGpu          true = TornadoVM kernels on GPU, false = pure-Java CPU path (Vector API)
 * @param maxTokens      context length reserved for the KV cache; also caps generated tokens
 * @param memoryMessages chat history window kept per session
 * @param queueCapacity  requests allowed to wait while the single inference thread is busy
 * @param requestTimeout how long a caller waits for a full answer before giving up
 */
@ConfigurationProperties("llm")
public record LlmProperties(
        Path modelPath,
        boolean onGpu,
        double temperature,
        double topP,
        int maxTokens,
        int seed,
        int memoryMessages,
        int queueCapacity,
        Duration requestTimeout) {
}
