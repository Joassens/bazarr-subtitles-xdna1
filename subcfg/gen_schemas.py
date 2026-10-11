"""Generate config.example/schemas/*.schema.json from one table (the single source of truth).

    python subcfg/gen_schemas.py config.example/schemas
"""
import json
import os
import sys

B, I, N, S, A = "boolean", "integer", "number", "string", "array"
# component -> rows of (dotted path, type, default, ENV name or None, title, description, extras)
TABLE = {
    "bridge": [
        ("backends.speech_urls", A, [], "WHISPER_ASR_URL", "Speech services",
         "whisper-asr-webservice compatible services, in order of preference. The first one that answers is used.", {"items": {"type": S}}),
        ("backends.prefer_native", B, False, "WHISPER_PREFER_NATIVE", "Prefer speech services",
         "Use them whenever reachable, not only when faster than the OpenAI-compatible fallback.", {}),
        ("backends.timeout_s", N, 600, "WHISPER_ASR_TIMEOUT", "Transcription timeout (s)",
         "A film on a slow backend can take hours.", {"minimum": 10}),
        ("backends.retry_offline_s", N, 60, "NATIVE_RETRY_SECS", "Retry offline backend after (s)", "", {"minimum": 0}),
        ("backends.probe_timeout_s", N, 3, "BACKEND_PROBE_TIMEOUT_SECS", "Health probe timeout (s)", "", {"minimum": 0.5}),
        ("llm.primary_url", S, "", "OPENAI_BASE_URL", "Primary LLM endpoint", "OpenAI-compatible base URL (ends in /v1).", {}),
        ("llm.fallback_url", S, "", "OPENAI_BASE_URL_FALLBACK", "Fallback LLM / whisper endpoint", "", {}),
        ("llm.model", S, "Qwen3-4B-Instruct-2507-GGUF-Q4_K_M", "TRANSLATE_LLM_MODEL", "Translation model (primary)", "", {}),
        ("llm.model_fallback", S, "", "TRANSLATE_LLM_MODEL_FALLBACK", "Translation model (fallback)", "Empty = same as the primary model.", {}),
        ("llm.dedicated_url", S, "", "TRANSLATE_LLM_BASE_URL", "Dedicated translation LLM",
         "When set and reachable it is used for every LLM call.", {}),
        ("llm.dedicated_model", S, "default", "TRANSLATE_LLM_MODEL_DEDICATED", "Dedicated LLM model name", "", {}),
        ("whisper.model", S, "whisper-1", "WHISPER_MODEL", "Whisper model name (OpenAI endpoints)", "", {}),
        ("whisper.translate_model", S, "", "WHISPER_TRANSLATE_MODEL", "Whisper model for translate tasks", "Empty = same as the model.", {}),
        ("whisper.initial_prompt", S, "", "WHISPER_INITIAL_PROMPT", "Static vocabulary hint", "Prepended to the automatic show context.", {}),
        ("whisper.forced_detect_language", S, "en", "FORCE_DETECTED_LANGUAGE_TO", "Language reported to Bazarr", "", {}),
        ("diarization.enabled", B, True, "WHISPER_DIARIZE", "Ask for speaker labels", "", {}),
        ("diarization.engine", S, "auto", "WHISPER_DIARIZE_ENGINE", "Diarization engine",
         "Which engine npusubs uses: auto (its own setting), pyannote (best, needs a Hugging Face token there) or embedding (token-free, coarser).",
         {"enum": ["auto", "pyannote", "embedding"]}),
        ("diarization.min_speakers", S, "", "WHISPER_MIN_SPEAKERS", "Minimum speakers", "", {}),
        ("diarization.max_speakers", S, "", "WHISPER_MAX_SPEAKERS", "Maximum speakers", "", {}),
        ("translation.batch_size", I, 5, "TRANSLATE_BATCH_SIZE", "Lines per translation batch", "", {"minimum": 1, "maximum": 50}),
        ("translation.context_lines", I, 6, "TRANSLATE_CONTEXT_LINES", "Earlier translated lines shown as context", "", {"minimum": 0, "maximum": 40}),
        ("translation.temperature", N, 0.2, "TRANSLATE_TEMPERATURE", "Temperature", "", {"minimum": 0, "maximum": 1.5}),
        ("translation.keep_honorifics", B, True, "TRANSLATE_KEEP_HONORIFICS", "Keep Japanese honorifics (-san, -kun)", "", {}),
        ("translation.bible", B, True, "TRANSLATE_BIBLE", "Scene bible pass", "Read the whole script first and write down names, terms, genders.", {}),
        ("translation.review", B, True, "TRANSLATE_REVIEW", "Review pass", "Conservative editor pass that only accepts real corrections.", {}),
        ("translation.review_batch", I, 10, "TRANSLATE_REVIEW_BATCH", "Lines per review batch", "", {"minimum": 1, "maximum": 40}),
        ("translation.bible_max_chars", I, 24000, "TRANSLATE_BIBLE_MAX_CHARS", "Max script size for the bible (chars)", "", {"minimum": 1000}),
        ("translation.debug", B, False, "TRANSLATE_DEBUG", "Log the bible and every review edit", "", {}),
        ("subtitles.max_line_length", I, 42, "MAX_LINE_LENGTH", "Max characters per line", "", {"minimum": 10, "maximum": 80}),
        ("subtitles.gap_split_s", N, 0.4, "GAP_SPLIT_SECS", "Silence that starts a new subtitle (s)", "", {"minimum": 0.1, "maximum": 5}),
        ("filters.phantom_start_max_s", N, 1.0, "PHANTOM_START_MAX_SECS", "Phantom cue window at start (s)", "0 disables.", {"minimum": 0}),
        ("filters.phantom_duration_max_s", N, 2.0, "PHANTOM_DURATION_MAX_SECS", "Phantom cue max duration (s)", "", {"minimum": 0}),
        ("filters.repetition_min_run", I, 4, "REPETITION_MIN_RUN_LENGTH", "Repetition loop: minimum repeats", "0 disables.", {"minimum": 0}),
        ("filters.repetition_max_period", I, 3, "REPETITION_MAX_PERIOD", "Repetition loop: pattern length", "", {"minimum": 1}),
        ("filters.global_duplicate_min", I, 3, "GLOBAL_DUPLICATE_MIN_COUNT", "Recurring hallucinated line: minimum count", "0 disables.", {"minimum": 0}),
        ("filters.skip_filename_pattern", S, r"(?i)\b(NC)?(OP|ED)\d{0,2}\b", "SKIP_FILENAME_PATTERN", "Skip files matching (OP/ED)", "", {}),
        ("audio.max_upload_mb", I, 24, "MAX_UPLOAD_MB", "Max upload per chunk (MB)", "", {"minimum": 1}),
        ("audio.opus_kbps", I, 24, "OPUS_BITRATE_KBPS", "Opus bitrate (kbps)", "", {"minimum": 6}),
        ("audio.silence_window_s", N, 20, "SILENCE_SEARCH_WINDOW_SECS", "Chunk split: silence search window (s)", "", {}),
        ("audio.silence_noise_db", S, "-35dB", "SILENCE_NOISE_DB", "Chunk split: silence threshold", "", {}),
        ("audio.silence_min_s", N, 0.3, "SILENCE_MIN_DURATION_SECS", "Chunk split: minimum silence (s)", "", {}),
        ("guard.heavy_window", S, "", "GUARD_HEAVY_WINDOW", "Allowed hours for jobs",
         "For example 01:00-07:00 (local time of the bridge; wraps past midnight). Jobs that arrive outside it wait. Empty = always.",
         {"pattern": r"^$|^\d{1,2}:\d{2}-\d{1,2}:\d{2}$"}),
        ("guard.min_free_host_ram_gb", N, 0, "GUARD_MIN_FREE_RAM_GB", "Minimum free host RAM for translations (GB)",
         "An iGPU takes its model memory from the host's RAM, so a big model can starve the host. Translation jobs wait while less is free. 0 = off. Needs the host RAM URL.",
         {"minimum": 0}),
        ("guard.host_ram_url", S, "", "GUARD_HOST_RAM_URL", "Host RAM URL",
         "A read-only endpoint on the machine that owns the memory, answering JSON with mem_available_gb and mem_total_gb (see tools/hostmem_agent.py).", {}),
        ("guard.max_llm_models", I, 1, "GUARD_MAX_LLM_MODELS", "Warn when more LLMs are loaded than this",
         "Shown on the dashboard; loading a second large model is what exhausted the host once.", {"minimum": 1}),
        ("guard.max_wait_s", N, 21600, "GUARD_MAX_WAIT_S", "Longest a job waits for the guard (s)",
         "After this the request fails (Bazarr will see an error) instead of waiting forever.", {"minimum": 60}),
        ("ui.enabled", B, True, "WEBUI_ENABLED", "Web UI", "Serve the web UI at /ui.", {}),
        ("ui.token", S, "", "WEBUI_TOKEN", "API token for scripts",
         "Optional. A script can send it as the header X-Token to use the web UI's JSON API. The web UI itself uses a user name "
         "and password chosen on first use. Empty = no API token.", {"x-secret": True}),
        ("ui.trusted_networks", A, [], "WEBUI_TRUSTED_NETWORKS", "Networks allowed to set the first password",
         "The first-time password page is offered to private and local addresses. If your LAN uses a different range, add it here "
         "(CIDR, one per line, for example 192.0.2.0/24).", {"items": {"type": S}}),
        ("arr.sonarr_url", S, "", "SONARR_URL", "Sonarr URL", "For show context in prompts.", {}),
        ("arr.sonarr_api_key", S, "", "SONARR_API_KEY", "Sonarr API key", "", {"x-secret": True}),
        ("arr.radarr_url", S, "", "RADARR_URL", "Radarr URL", "", {}),
        ("arr.radarr_api_key", S, "", "RADARR_API_KEY", "Radarr API key", "", {"x-secret": True}),
        ("arr.cache_ttl_s", N, 1800, "MEDIA_CACHE_TTL_SECS", "Show-context cache (s)", "", {"minimum": 0}),
    ],
    "npusubs": [
        ("model", S, "large-v3-turbo", "NPUSUBS_MODEL", "Whisper model", "", {"x-restart": True}),
        ("threads", I, 8, "NPUSUBS_THREADS", "CPU threads for the decoder", "", {"minimum": 1, "maximum": 64, "x-restart": True}),
        ("queue_wait_s", N, 21600, "NPUSUBS_QUEUE_WAIT", "Max wait for the NPU (s)", "One job at a time.", {"minimum": 1}),
        ("beam", I, 5, "NPUSUBS_BEAM", "Beam width", "1 = greedy (fast), 5 = accurate (about 4-5x slower).", {"minimum": 1, "maximum": 10}),
        ("vad.enabled", B, True, "NPUSUBS_VAD", "Skip silence (VAD)", "", {}),
        ("vad.threshold", N, 0.35, None, "VAD threshold", "Lower keeps more.", {"minimum": 0.05, "maximum": 0.95}),
        ("vad.min_silence_ms", I, 600, None, "VAD minimum silence (ms)", "", {"minimum": 50}),
        ("vad.speech_pad_ms", I, 400, None, "VAD padding (ms)", "", {"minimum": 0}),
        ("vad.min_speech_ms", I, 150, None, "VAD minimum speech (ms)", "", {"minimum": 20}),
        ("vad.join_gap_s", N, 1.5, None, "Merge speech closer than (s)", "", {"minimum": 0}),
        ("vad.music.enabled", B, True, "NPUSUBS_MUSIC", "Keep and transcribe songs", "Sustained loud non-speech counts as music.", {}),
        ("vad.music.db", N, -33.0, None, "Music loudness threshold (dBFS)", "", {}),
        ("vad.music.min_s", N, 8.0, None, "Music minimum length (s)", "", {"minimum": 1}),
        ("diarization.default", B, False, "NPUSUBS_DIARIZE", "Diarize by default", "Needs HF_TOKEN and the gated model terms.", {}),
        ("diarization.parallel", B, False, "NPUSUBS_DIARIZE_PARALLEL", "Overlap diarization with transcription", "Higher peak memory.", {}),
        ("diarization.model", S, "pyannote/speaker-diarization-community-1", "NPUSUBS_DIARIZE_MODEL", "Diarization model", "", {}),
        ("diarization.engine", S, "auto", "NPUSUBS_DIARIZE_ENGINE", "Diarization engine",
         "pyannote = best, needs a Hugging Face token. embedding = token-free and about 15x cheaper, but coarser (no overlapped speech, "
         "cannot split a very fast back-and-forth). auto = pyannote when it is usable, otherwise embedding.",
         {"enum": ["auto", "pyannote", "embedding"]}),
        ("diarization.embedding.threshold", N, 0.8, None, "Embedding engine: how different two voices must be",
         "Cosine distance at which clusters stay apart. Lower = more speakers, higher = fewer. 0.8 gave the right number of speakers in tests.",
         {"minimum": 0.3, "maximum": 1.2}),
        ("diarization.embedding.window_s", N, 1.5, None, "Embedding engine: window (s)", "Length of the audio piece that gets one voice print.", {"minimum": 0.8, "maximum": 4}),
        ("diarization.embedding.hop_s", N, 0.5, None, "Embedding engine: step (s)", "How far the window moves; also the finest speaker change it can see.", {"minimum": 0.2, "maximum": 2}),
        ("diarization.embedding.min_speaker_s", N, 3.0, None, "Embedding engine: shortest speaker (s)", "A voice that speaks less than this in total is merged into the nearest other one.", {"minimum": 0}),
        ("subtitles.max_line", I, 42, "NPUSUBS_MAX_LINE", "Max display columns per line", "CJK counts double.", {"minimum": 10, "maximum": 80}),
        ("subtitles.gap_split_s", N, 0.5, "NPUSUBS_GAP_SPLIT", "Silence that starts a new cue (s)", "", {"minimum": 0.1}),
        ("subtitles.speaker_style", S, "dash", "NPUSUBS_SPEAKER_STYLE", "Speaker style", "", {"enum": ["none", "dash", "label"]}),
        ("subtitles.min_duration_s", N, 0.8, None, "Minimum cue duration (s)", "", {"minimum": 0.2}),
        ("subtitles.max_duration_s", N, 7.0, None, "Maximum cue duration (s)", "", {"minimum": 1}),
    ],
    "settings": [
        ("resource_guard.min_free_host_ram_gb", N, 12, None, "Minimum free host RAM (GB)",
         "Refuse to activate a model or backend that would leave less than this.", {"minimum": 0}),
        ("resource_guard.max_concurrent_llm_models", I, 1, None, "Max LLM models loaded at once", "", {"minimum": 1}),
        ("resource_guard.heavy_jobs_window", S, "", None, "Window for heavy jobs (HH:MM-HH:MM)", "Empty = any time.", {}),
    ],
}


def build(rows):
    root = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}}
    for path, typ, default, env, title, desc, extra in rows:
        node = root
        parts = path.split(".")
        for p in parts[:-1]:
            node = node["properties"].setdefault(p, {"type": "object", "properties": {}})
        leaf = {"type": typ, "default": default, "title": title}
        if desc:
            leaf["description"] = desc
        if env:
            leaf["x-env"] = env
        leaf.update(extra)
        node["properties"][parts[-1]] = leaf
    return root


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "config.example/schemas"
    os.makedirs(out, exist_ok=True)
    for comp, rows in TABLE.items():
        schema = build(rows)
        schema["$id"] = comp
        with open(os.path.join(out, f"{comp}.schema.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"{comp}: {len(rows)} instellingen")
