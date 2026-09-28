package id.poc.inappai.agent;

import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;

/** Conversational agent with per-session memory. Implemented by LangChain4j {@code AiServices}. */
public interface ChatAgent {

    @SystemMessage(fromResource = "prompts/agent-system.txt")
    String chat(@MemoryId String sessionId, @UserMessage String message);

    @SystemMessage(fromResource = "prompts/agent-system.txt")
    TokenStream chatStream(@MemoryId String sessionId, @UserMessage String message);
}
