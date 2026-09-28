package id.poc.inappai.llm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;

class GatedStreamingChatModelTest {

    private final ExecutorService executor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(1), new ThreadPoolExecutor.AbortPolicy());

    @AfterEach
    void shutdown() {
        executor.shutdownNow();
    }

    /** Fake GPU model that records how many generations overlap. */
    static class FakeModel implements StreamingChatModel {
        final AtomicInteger active = new AtomicInteger();
        final AtomicInteger maxActive = new AtomicInteger();
        volatile CountDownLatch release = new CountDownLatch(0);

        @Override
        public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            maxActive.accumulateAndGet(active.incrementAndGet(), Math::max);
            try {
                release.await(5, TimeUnit.SECONDS);
                handler.onPartialResponse("hi");
                handler.onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage.from("hi")).build());
            } catch (InterruptedException e) {
                handler.onError(e);
            } finally {
                active.decrementAndGet();
            }
        }
    }

    private static ChatRequest request() {
        return ChatRequest.builder().messages(List.of(UserMessage.from("ping"))).build();
    }

    @Test
    void generationsNeverOverlap() {
        FakeModel fake = new FakeModel();
        BlockingChatModel model = new BlockingChatModel(
                new GatedStreamingChatModel(fake, executor, Duration.ofSeconds(5)), Duration.ofSeconds(5));

        List<CompletableFuture<ChatResponse>> calls = List.of(
                CompletableFuture.supplyAsync(() -> model.chat(request())),
                CompletableFuture.supplyAsync(() -> model.chat(request())));

        calls.forEach(c -> assertThat(c.join().aiMessage().text()).isEqualTo("hi"));
        assertThat(fake.maxActive.get()).isEqualTo(1);
    }

    @Test
    void rejectsWhenQueueIsFull() throws Exception {
        FakeModel fake = new FakeModel();
        fake.release = new CountDownLatch(1);
        GatedStreamingChatModel gated = new GatedStreamingChatModel(fake, executor, Duration.ofSeconds(5));
        BlockingChatModel model = new BlockingChatModel(gated, Duration.ofSeconds(5));

        CompletableFuture<ChatResponse> running = CompletableFuture.supplyAsync(() -> model.chat(request()));
        while (fake.active.get() == 0) {
            Thread.sleep(10);
        }
        CompletableFuture<ChatResponse> queued = CompletableFuture.supplyAsync(() -> model.chat(request()));
        while (((ThreadPoolExecutor) executor).getQueue().isEmpty()) {
            Thread.sleep(10);
        }

        assertThatThrownBy(() -> model.chat(request())).isInstanceOf(InferenceBusyException.class);

        fake.release.countDown();
        assertThat(running.join().aiMessage().text()).isEqualTo("hi");
        assertThat(queued.join().aiMessage().text()).isEqualTo("hi");
    }
}
