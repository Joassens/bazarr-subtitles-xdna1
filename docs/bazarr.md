# Using npusubs with Bazarr

## Recommended: through the bridge

Bazarr → Settings → Providers → **Whisper** (the `whisperai` provider):

* Endpoint: `http://<host>:9500` (the bridge; it passes the audio on to npusubs and translates with your LLM when asked)
* Response timeout / transcription timeout: generous. On the NPU a 24 minute episode takes about 6-8 minutes with greedy
  decoding, 30-40 minutes with beam search and speaker recognition; set hours, not minutes.
* For a library with several languages, make a Bazarr **language profile** (Settings → Languages), for example "Anime JP > EN"
  with the original language (audio only) and the language you want, and attach the series to it.

## Direct to npusubs (no translation, no web UI)

Same provider, but with the endpoint `http://<host>:9091`. Use this when you only want subtitles in the audio's own language.

* Same timeout advice as above.
* Enable "Pass video filename to Whisper" if you want to keep it.

Bazarr sends raw 16 kHz mono PCM with `encode=false`; npusubs accepts that, and `output=srt`.

## With translation (for example Japanese audio → English or Dutch subtitles)

Whisper's own translate task is weak with large-v3-turbo. A common setup is a small bridge between Bazarr
and the speech service that transcribes in the source language and translates with an LLM. The `/asr` JSON
(`output=json`) contains word timings and, with diarization, a `speaker` per word, so such a bridge can
split subtitles on speaker changes. The [`bridge/`](../bridge/README.md) in this repository (Bazarr → bridge → npusubs → LLM) is a modified copy of
[`mccloud/bazarr-openai-whisperbridge`](https://hub.docker.com/r/mccloud/bazarr-openai-whisperbridge); it needs
only the speech service URL (`backends.speech_urls`, or `WHISPER_ASR_URL=http://<host>:9091`) to talk to npusubs.

`WHISPER_ASR_URL` accepts a comma separated list in this bridge: put a faster GPU service first and
npusubs second, and the first one that answers is used.

## API summary

* `POST /asr` (multipart `audio_file`; query: `language`, `task`, `initial_prompt`, `output=txt|srt|vtt|tsv|json`,
  `encode=true|false`, `vad_filter`, `diarize`, `min_speakers`, `max_speakers`)
* `POST /detect-language` (multipart `audio_file`, `encode`)
* `POST /v1/audio/transcriptions` (OpenAI style; `response_format=json|verbose_json|text|srt|vtt`)
* `GET /health`
