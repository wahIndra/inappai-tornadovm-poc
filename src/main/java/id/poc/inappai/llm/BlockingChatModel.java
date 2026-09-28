package id.poc.inappai.llm;

import java.time.Duration;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

/**
 * Blocking {@link ChatModel} view over the gated streaming model, so sync and streaming calls share one
 * set of GPU weights and one inference queue instead of loading the model twice.
 */
public class BlockingChatModel implements ChatModel {

    private final StreamingChatModel streaming;
    private final Duration requestTimeout;

    public BlockingChatModel(StreamingChatModel streaming, Duration requestTimeout) {
        this.streaming = streaming;
        this.requestTimeout = requestTimeout;
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        CompletableFuture<ChatResponse> result = new CompletableFuture<>();
        streaming.chat(request, new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(String partialResponse) {
            }

            @Override
            public void onCompleteResponse(ChatResponse completeResponse) {
                result.complete(completeResponse);
            }

            @Override
            public void onError(Throwable error) {
                result.completeExceptionally(error);
            }
        });
        try {
            return result.get(requestTimeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            if (e.getCause() instanceof RuntimeException re) {
                throw re;
            }
            throw new IllegalStateException("LLM generation failed", e.getCause());
        } catch (TimeoutException e) {
            throw new IllegalStateException("LLM did not answer within " + requestTimeout, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for LLM", e);
        }
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return streaming.defaultRequestParameters();
    }

    @Override
    public Set<Capability> supportedCapabilities() {
        return streaming.supportedCapabilities();
    }
}
