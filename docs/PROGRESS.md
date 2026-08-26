# Progress

## Completed Milestones

- Project and virtual environment setup completed
- Core dependencies installed
- Modular project structure created
- LLM and search provider abstractions created
- Mock LLM and mock search implemented
- Research Agent implemented
- LangGraph research workflow implemented
- Automated tests created
- All 44 tests passing
- Mock end-to-end Research Agent successfully validated
- Real Wikipedia provider integrated successfully
- Real Gemini provider integrated successfully
- Retry/backoff handling added for transient Gemini errors
- Optional fallback Gemini model support added
- Live Research Agent demo succeeded
- Final structured ResearchResult produced from real sources
- 92/92 tests passing
- Research Agent MVP marked complete
- Script Agent implemented (ResearchResult → structured ScriptResult)
- ScriptAgent reuses the existing LLMProvider abstraction (no direct Gemini coupling)
- LangGraph script workflow implemented, mirroring the research workflow
- Script Agent unit tests added using mock LLM providers (no real Gemini calls)
- Research Agent -> Script Agent demo/runner validated end to end (mock providers)
- 129/129 tests passing
- Script Agent MVP marked complete
- Research Agent MVP complete
- Script Agent MVP complete
- Real Research → Script LangGraph pipeline implemented
- Real Wikipedia search used successfully
- Real Gemini LLM used successfully
- Gemini retry/backoff and fallback model handling validated
- Primary model currently configured as gemini-3.5-flash-lite for reliable MVP testing
- Fallback model configured as gemini-3.6-flash
- Live end-to-end Research → Script pipeline completed successfully
- Current automated test count: 139 passed
- Voice Generation Service implemented as a deterministic service (not an LLM agent)
- VoiceProvider abstraction created, with Mock and free Edge TTS implementations
- VoiceService extracts and orders narration (hook → introduction → sections → conclusion → call_to_action) while excluding metadata (sources, research notes, headings, visual notes)
- Real Edge TTS narration audio generated successfully, saved to the gitignored `output/audio/` directory
- Voice provider selection added via `VOICE_PROVIDER`/`VOICE_NAME` settings, mirroring the LLM/search provider pattern
- Voice Service unit tests added using a mock voice provider and a monkeypatched Edge TTS client (no real network calls)
- Current automated test count: 175 passed
- Voice Generation Service MVP marked complete
- Voice Service MVP completed using Edge TTS
- Real MP3 generation validated locally
- Script narration repetition issue identified and resolved
- Root cause was repetitive ScriptAgent section generation
- ScriptAgent now enforces distinct sections with retry-before-drop validation
- VoiceService retains deduplication as a safety layer
- Final regenerated narration kept 5/5 distinct sections
- Manual audio review confirmed no repeated narration
- 188/188 tests passing
- Research → Script → Voice LangGraph pipeline completed
- Real Wikipedia + Gemini + Edge TTS validated in one run
- VoiceResult stored in pipeline state
- Failure short-circuit behavior tested
- Real MP3 generated successfully
- 190/190 tests passing

## Current Next Milestone

Visual/Media generation service.
