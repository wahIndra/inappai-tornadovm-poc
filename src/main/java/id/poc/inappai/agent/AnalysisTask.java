package id.poc.inappai.agent;

/** Analysis tasks exposed by {@code POST /api/analyze}; each maps to an instruction for {@link TextAnalyst}. */
public enum AnalysisTask {

    SUMMARIZE("Summarize the text in at most 3 sentences, in the same language as the text."),
    SENTIMENT("Classify the overall sentiment as exactly one word: POSITIVE, NEGATIVE or NEUTRAL. Then give a one-sentence reason."),
    KEYWORDS("List the 5 most important keywords or keyphrases, comma-separated, nothing else."),
    ENTITIES("List the named entities (people, organizations, places, dates) as lines in the form 'TYPE: value'."),
    TRANSLATE_ID("Translate the text into natural Bahasa Indonesia."),
    TRANSLATE_EN("Translate the text into natural English.");

    private final String instruction;

    AnalysisTask(String instruction) {
        this.instruction = instruction;
    }

    public String instruction() {
        return instruction;
    }
}
