package id.poc.inappai.agent;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

/** Stateless one-shot text analysis. Implemented by LangChain4j {@code AiServices}. */
public interface TextAnalyst {

    @SystemMessage("You are a precise text analysis engine. Follow the instruction exactly and output only the result.")
    @UserMessage("""
            Instruction: {{instruction}}

            Text:
            \"\"\"
            {{text}}
            \"\"\"
            """)
    String analyze(@V("instruction") String instruction, @V("text") String text);
}
