package id.poc.inappai.config;

import java.nio.file.Files;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.gpullama3.GPULlama3StreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import dev.langchain4j.store.memory.chat.InMemoryChatMemoryStore;
import id.poc.inappai.agent.ChatAgent;
import id.poc.inappai.agent.TextAnalyst;
import id.poc.inappai.llm.BlockingChatModel;
import id.poc.inappai.llm.GatedStreamingChatModel;

@Configuration
public class LlmConfig {

    private static final Logger log = LoggerFactory.getLogger(LlmConfig.class);

    /** One thread owns the GPU model: loading and every generation run here, one at a time. */
    @Bean(destroyMethod = "shutdownNow")
    ExecutorService inferenceExecutor(LlmProperties props) {
        return new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(props.queueCapacity()),
                Thread.ofPlatform().name("llm-inference").daemon(true).factory(),
                new ThreadPoolExecutor.AbortPolicy());
    }

    @Bean(destroyMethod = "close")
    GPULlama3StreamingChatModel gpuLlama3(LlmProperties props, ExecutorService inferenceExecutor)
            throws InterruptedException {
        if (!Files.isRegularFile(props.modelPath())) {
            throw new IllegalStateException("Model file not found: " + props.modelPath().toAbsolutePath()
                    + " - download one with .\\scripts\\download-model.ps1 or set LLM_MODEL_PATH");
        }
        log.info("Loading {} ({}) ...", props.modelPath().getFileName(), props.onGpu() ? "GPU / TornadoVM" : "CPU");
        long start = System.nanoTime();
        try {
            GPULlama3StreamingChatModel model = inferenceExecutor.submit(() -> GPULlama3StreamingChatModel.builder()
                    .modelPath(props.modelPath())
                    .onGPU(props.onGpu())
                    .temperature(props.temperature())
                    .topP(props.topP())
                    .maxTokens(props.maxTokens())
                    .seed(props.seed())
                    .build()).get();
            log.info("Model ready in {} ms", TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start));
            return model;
        } catch (ExecutionException e) {
            throw new IllegalStateException("Failed to load model " + props.modelPath(), e.getCause());
        }
    }

    @Bean
    StreamingChatModel streamingChatModel(GPULlama3StreamingChatModel gpuLlama3, ExecutorService inferenceExecutor,
            LlmProperties props) {
        return new GatedStreamingChatModel(gpuLlama3, inferenceExecutor, props.requestTimeout());
    }

    @Bean
    ChatModel chatModel(StreamingChatModel streamingChatModel, LlmProperties props) {
        return new BlockingChatModel(streamingChatModel, props.requestTimeout());
    }

    @Bean
    ChatMemoryStore chatMemoryStore() {
        return new InMemoryChatMemoryStore();
    }

    @Bean
    ChatAgent chatAgent(ChatModel chatModel, StreamingChatModel streamingChatModel, ChatMemoryStore chatMemoryStore,
            LlmProperties props) {
        ChatMemoryProvider memoryProvider = sessionId -> MessageWindowChatMemory.builder()
                .id(sessionId)
                .maxMessages(props.memoryMessages())
                .chatMemoryStore(chatMemoryStore)
                .build();
        return AiServices.builder(ChatAgent.class)
                .chatModel(chatModel)
                .streamingChatModel(streamingChatModel)
                .chatMemoryProvider(memoryProvider)
                .build();
    }

    @Bean
    TextAnalyst textAnalyst(ChatModel chatModel) {
        return AiServices.builder(TextAnalyst.class)
                .chatModel(chatModel)
                .build();
    }
}
