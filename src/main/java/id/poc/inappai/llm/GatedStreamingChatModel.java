package id.poc.inappai.llm;

import java.time.Duration;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

/**
 * Serializes access to the GPULlama3 model.
 *
 * <p>GPULlama3 keeps its KV cache and TornadoVM execution plan inside the model instance, so it must never
 * run two generations at once. Every request is handed to a single-thread executor with a bounded queue;
 * the thread stays occupied until the generation completes, and overflow is rejected with
 * {@link InferenceBusyException}.
 */
public class GatedStreamingChatModel implements StreamingChatModel {

    private static final Logger log = LoggerFactory.getLogger(GatedStreamingChatModel.class);

    private final StreamingChatModel delegate;
    private final ExecutorService inferenceExecutor;
    private final Duration requestTimeout;

    public GatedStreamingChatModel(StreamingChatModel delegate, ExecutorService inferenceExecutor, Duration requestTimeout) {
        this.delegate = delegate;
        this.inferenceExecutor = inferenceExecutor;
        this.requestTimeout = requestTimeout;
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        try {
            inferenceExecutor.execute(() -> generate(request, handler));
        } catch (RejectedExecutionException e) {
            handler.onError(new InferenceBusyException());
        }
    }

    private void generate(ChatRequest request, StreamingChatResponseHandler handler) {
        CompletableFuture<Void> done = new CompletableFuture<>();
        try {
            delegate.chat(request, new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String partialResponse) {
                    handler.onPartialResponse(partialResponse);
                }

                @Override
                public void onCompleteResponse(ChatResponse completeResponse) {
                    try {
                        handler.onCompleteResponse(completeResponse);
                    } finally {
                        done.complete(null);
                    }
                }

                @Override
                public void onError(Throwable error) {
                    try {
                        handler.onError(error);
                    } finally {
                        done.complete(null);
                    }
                }
            });
            // Hold the inference thread until this generation is finished, even if the delegate is async.
            done.get(requestTimeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            log.warn("Generation exceeded {}; releasing inference thread", requestTimeout);
            handler.onError(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            handler.onError(e);
        } catch (Exception e) {
            if (!done.isDone()) {
                handler.onError(e);
            }
        }
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return delegate.defaultRequestParameters();
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return delegate.supportedCapabilities();
    }
}
