package id.poc.inappai.web;

import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import id.poc.inappai.agent.AnalysisTask;
import id.poc.inappai.agent.ChatAgent;
import id.poc.inappai.agent.TextAnalyst;
import id.poc.inappai.config.LlmProperties;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

@RestController
@RequestMapping("/api")
public class AiController {

    private static final Logger log = LoggerFactory.getLogger(AiController.class);

    public record ChatRequest(String sessionId, @NotBlank @Size(max = 4000) String message) {
    }

    public record ChatReply(String sessionId, String reply, long elapsedMs) {
    }

    public record AnalyzeRequest(@NotNull AnalysisTask task, @NotBlank @Size(max = 8000) String text) {
    }

    public record AnalyzeReply(AnalysisTask task, String result, long elapsedMs) {
    }

    private final ChatAgent chatAgent;
    private final TextAnalyst textAnalyst;
    private final ChatMemoryStore chatMemoryStore;
    private final LlmProperties props;
    private final ThreadPoolExecutor inferenceExecutor;

    public AiController(ChatAgent chatAgent, TextAnalyst textAnalyst, ChatMemoryStore chatMemoryStore,
            LlmProperties props, ExecutorService inferenceExecutor) {
        this.chatAgent = chatAgent;
        this.textAnalyst = textAnalyst;
        this.chatMemoryStore = chatMemoryStore;
        this.props = props;
        this.inferenceExecutor = (ThreadPoolExecutor) inferenceExecutor;
    }

    @GetMapping("/info")
    public Map<String, Object> info() {
        return Map.of(
                "model", props.modelPath().getFileName().toString(),
                "backend", props.onGpu() ? "GPU (TornadoVM)" : "CPU (pure Java)",
                "maxTokens", props.maxTokens(),
                "memoryMessages", props.memoryMessages(),
                "requestsAhead", requestsAhead(),
                "tasks", AnalysisTask.values());
    }

    @PostMapping("/chat")
    public ChatReply chat(@Valid @RequestBody ChatRequest request) {
        String sessionId = sessionIdOrNew(request.sessionId());
        long start = System.currentTimeMillis();
        String reply = chatAgent.chat(sessionId, request.message());
        return new ChatReply(sessionId, reply, System.currentTimeMillis() - start);
    }

    /**
     * Server-Sent Events: {@code session} (id), {@code queued} (requests ahead of this one), many
     * {@code token} events, then {@code done} with timing, or {@code error}.
     */
    @PostMapping(path = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chatStream(@Valid @RequestBody ChatRequest request) throws IOException {
        String sessionId = sessionIdOrNew(request.sessionId());
        SseEmitter emitter = new SseEmitter(props.requestTimeout().plus(Duration.ofSeconds(5)).toMillis());
        emitter.send(SseEmitter.event().name("session").data(sessionId));
        emitter.send(SseEmitter.event().name("queued").data(requestsAhead()));

        long start = System.currentTimeMillis();
        AtomicInteger chunks = new AtomicInteger();
        chatAgent.chatStream(sessionId, request.message())
                .onPartialResponse(token -> {
                    chunks.incrementAndGet();
                    send(emitter, "token", token);
                })
                .onCompleteResponse(response -> {
                    long elapsed = System.currentTimeMillis() - start;
                    send(emitter, "done", Map.of("elapsedMs", elapsed, "chunks", chunks.get()));
                    emitter.complete();
                })
                .onError(error -> {
                    log.warn("Streaming chat failed: {}", error.toString());
                    send(emitter, "error", String.valueOf(error.getMessage()));
                    emitter.complete();
                })
                .start();
        return emitter;
    }

    @DeleteMapping("/chat/{sessionId}")
    public Map<String, String> clearSession(@PathVariable String sessionId) {
        chatMemoryStore.deleteMessages(sessionId);
        return Map.of("sessionId", sessionId, "status", "cleared");
    }

    @PostMapping("/analyze")
    public AnalyzeReply analyze(@Valid @RequestBody AnalyzeRequest request) {
        long start = System.currentTimeMillis();
        String result = textAnalyst.analyze(request.task().instruction(), request.text());
        return new AnalyzeReply(request.task(), result.strip(), System.currentTimeMillis() - start);
    }

    /** Generations running or waiting on the single inference thread. */
    private int requestsAhead() {
        return inferenceExecutor.getActiveCount() + inferenceExecutor.getQueue().size();
    }

    private static String sessionIdOrNew(String sessionId) {
        return sessionId == null || sessionId.isBlank() ? UUID.randomUUID().toString() : sessionId;
    }

    private static void send(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data));
        } catch (IOException | IllegalStateException e) {
            // Client disconnected; generation keeps running on the inference thread until it finishes.
            log.debug("SSE send failed: {}", e.toString());
        }
    }
}
