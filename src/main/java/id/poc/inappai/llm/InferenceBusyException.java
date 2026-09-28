package id.poc.inappai.llm;

/** Thrown when the inference queue is full; mapped to HTTP 503 so clients can retry. */
public class InferenceBusyException extends RuntimeException {

    public InferenceBusyException() {
        super("LLM is busy, inference queue is full. Retry shortly.");
    }
}
