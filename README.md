# Bazarr subtitles with XDNA1 (npusubs)

Subtitle-grade speech-to-text on the **AMD XDNA 1 NPU** (Ryzen 7 8845HS / 7840HS "Hawk Point" / "Phoenix"),
with the API Bazarr already speaks. **Accuracy over speed:** beam search, songs and endings are subtitled,
and it is meant to run as a slow background job.

The Whisper *encoder* runs on the NPU; decoding, word timestamps, voice activity detection, music handling,
speaker diarization and subtitle formatting run on the CPU. The point is not raw speed: it is to run
Whisper on a part of the chip that is otherwise idle, so the iGPU stays free for an LLM (for example
subtitle translation) running at the same time.

> **Status: early.** It works end to end on one machine (Ryzen 7 8845HS, Linux 7.0, NPU firmware 1.5.5.391)
> and is wired into Bazarr through a bridge. Treat it as a working prototype. Issues and results from other
> machines are very welcome.

## What you get

| | |
|---|---|
| Whisper large-v3-turbo with the encoder on the NPU | via [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna) (MIT) |
| Word-level timestamps, language detection, prompts | through stock `openai-whisper`: the NPU encoder replaces `model.encoder` |
| Silero VAD that skips silence and packs speech into 30 s windows | roughly 2-3x less NPU work |
| **Music / song handling** | speech VADs cannot see singing under music, so sustained loud non-speech (openings, endings) is kept and transcribed as lyrics |
| Hallucination filters | repetition loops, recurring lines, phantom first cue, empty cues |
| Subtitle formatting | max line length (CJK double width), balanced 2-line cues, gap/speaker/sentence splits, SRT / VTT / TXT / TSV / JSON |
| **Speaker diarization** (optional) | pyannote, new cue on speaker change, `- A / - B` dash style or `[SPEAKER_00]` labels |
| `whisper-asr-webservice` API | `POST /asr`, `POST /detect-language`: works with Bazarr's Whisper provider |
| OpenAI-compatible API | `POST /v1/audio/transcriptions` (`json`, `verbose_json` with words, `srt`, `vtt`, `text`) |

## Requirements

* AMD Ryzen with an **XDNA 1 NPU** (Phoenix / Hawk Point: 5 columns). *XDNA 2 (Strix) is not what this targets*:
  use [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) / Lemonade for that generation.
* Linux **6.14+** with the in-tree `amdxdna` driver, `/dev/accel/accel0` present, recent NPU firmware.
* Docker. About 12 GB for the image, 8-9 GiB RAM while running.
* First start downloads the Whisper checkpoint (~1.6 GB) and compiles the NPU overlays (minutes, then cached).

## How the pieces fit

This repository is **one stack of two services** that start together with one `docker compose up`:

```
 Bazarr ──> bridge (:9500) ──> npusubs (:9091)   speech to text on the NPU (or any whisper-asr-webservice)
              │    │                  ▲
              │    └──> LLM (OpenAI-compatible, for example Lemonade)   translation: bible + review pass
              │
              └── web UI at /ui   dashboard, settings editor, profiles, playground, request detail
                        │
                        └── config/ (JSON + secrets.env + history/)   read by the bridge and by npusubs, hot reloaded
```

| Piece | Folder | What it does | Who talks to it |
|---|---|---|---|
| **bridge** | [`bridge/`](bridge/README.md) | The address you give Bazarr. Hands the audio to a speech service, translates with your LLM (scene bible + review pass), per-show profiles, the web UI. | Bazarr, you (browser) |
| **npusubs** | [`npusubs/`](npusubs/) | The speech engine: Whisper large-v3-turbo with the encoder on the NPU, silence/music handling, speaker recognition, subtitle formatting. | the bridge |
| **your LLM** | (not included) | Any OpenAI-compatible server, for example [Lemonade](https://github.com/lemonade-sdk/lemonade) or llama.cpp. Only needed for translation. | the bridge |

Bazarr only ever talks to the **bridge**. They are two containers on purpose: npusubs needs the NPU driver and a heavy
Python environment, the bridge is light and works with any `whisper-asr-webservice` too, and you can restart or update the
bridge without interrupting a transcription that is running.

**What you need, depending on what you want:**

| I want | Run | Also needed |
|---|---|---|
| Everything (an XDNA1 NPU machine) | `docker compose up -d` | nothing else; an LLM URL for translation |
| Only subtitles in the audio's own language, no web UI | `docker compose up -d npusubs`, Bazarr → `:9091` (works without the bridge) | |
| Translation + web UI, **no NPU** | `docker compose up -d bridge` and point it at another speech service | see [No NPU?](#no-npu) |
| Best speaker recognition | | a free Hugging Face account that accepted a licence: [docs/huggingface.md](docs/huggingface.md) |

## Bridge and web UI

[`bridge/`](bridge/README.md) holds a modified copy of McCloudS/bazarr-openai-whisperbridge (Apache-2.0): several
speech backends with failover, LLM translation with a scene "bible" and a conservative review pass, and
speaker-aware subtitles. It also serves a small **web UI** (`/ui`, no build step) to watch and tune everything:

* **Dashboard**: health of every service, what the NPU is doing, the last requests; open a request to compare the
  draft and the reviewed translation and download the SRT. A **resource guard** can hold jobs outside allowed hours or
  when the host is low on RAM.
* **Configuration**: every setting of the bridge and npusubs with its description, edited with validation; changes apply
  within seconds, every save is a snapshot you can compare and restore (**History**).
* **Profiles**: different settings per show, path or language.
* **Playground**: run the same subtitles or clip with different settings and compare the results side by side.

You sign in with a user name and password: the first time you open `/ui` it asks you to choose the password (default
user `admin`), so there is nothing to set up beforehand. It needs the config folder mounted read-write. See
[docs/web-ui-getting-started.md](docs/web-ui-getting-started.md),
[bridge/README.md](bridge/README.md) and [docs/webui-design.md](docs/webui-design.md).

## Quick start

```bash
# 1. build the NPU base image once (XRT + MLIR-AIE + torch + openai-whisper)
git clone https://github.com/drakosha/whisper-xdna && cd whisper-xdna
docker build --build-arg WITH_REFERENCE=1 -t npu-whisper:latest .

# 2. get this repository and a config folder
git clone https://github.com/Joassens/bazarr-subtitles-xdna1 && cd bazarr-subtitles-xdna1
cp -r config.example config
cp config/secrets.env.example config/secrets.env && chmod 600 config/secrets.env

# 3. build and start the whole stack (bridge + npusubs)
RENDER_GID=$(getent group render | cut -d: -f3) docker compose up -d
curl localhost:9091/health            # status: ok   (npusubs)
```

4. **Open `http://<host>:9500/ui`** and choose a password (the first visit asks for it; see
   [docs/web-ui-getting-started.md](docs/web-ui-getting-started.md)). Under *Configuration* fill in `llm.primary_url`
   (your OpenAI-compatible LLM, ending in `/v1`) and the translation model if you want translated subtitles.
5. **In Bazarr** → Settings → Providers → Whisper: endpoint `http://<host>:9500` (see below).

Try the speech engine on its own, without Bazarr:
`curl -F audio_file=@episode.mkv "localhost:9091/asr?language=ja&output=srt" > episode.ja.srt`

The npusubs container needs `IPC_LOCK` and an unlimited `memlock` (XRT locks its device buffers). Inside an LXC
container Docker may refuse `memlock: -1`; raise the hard limit of `containerd`/`dockerd` from the host
(`prlimit --pid <pid> --memlock=unlimited:unlimited`) or set `lxc.prlimit.memlock: unlimited`.

Speaker recognition with the best engine needs a Hugging Face account: [docs/huggingface.md](docs/huggingface.md). Everything
else works without one.

### No NPU?

The bridge does not need the NPU. Start only the bridge, `docker compose up -d bridge`, then in the web UI (*Configuration* →
`backends.speech_urls`) replace `http://npusubs:9091` with any `whisper-asr-webservice` you have (faster-whisper on a GPU, for
example). You keep the translation pipeline, profiles and web UI. npusubs itself is specific to the XDNA1 NPU.

## Using it with Bazarr

Bazarr's **Whisper** provider (`whisperai`) talks `whisper-asr-webservice`. Point it at the **bridge**,
`http://<host>:9500`, and give it a generous timeout (a 24 minute episode takes 30-40 minutes with beam search and
speaker recognition; set the response/transcription timeouts to hours). The bridge then calls npusubs and, for
"translate" requests, your LLM. npusubs also speaks that API itself (`http://<host>:9091`), so Bazarr can use it
directly when you do not need translation or the web UI.
See [docs/bazarr.md](docs/bazarr.md).

## Configuration

Settings live in JSON files in a config folder (see [config.example](config.example/README.md)): copy it to
`./config` (or set `NPUSUBS_CONFIG_DIR`) and edit `config/npusubs.json`. Defaults, allowed values and descriptions come
from the schemas in `config/schemas/`; changes apply within a few seconds without a restart (only the model and
thread count need one), and a broken file never takes the service down (`/health` reports the problem and the
last good configuration stays active). The environment variables below still work and override the file when set.

### Environment variables (optional overrides)

| Variable | Default | |
|---|---|---|
| `NPUSUBS_MODEL` | `large-v3-turbo` | `large-v3` also works |
| `NPUSUBS_THREADS` | `8` | CPU threads for the decoder |
| `NPUSUBS_BEAM` | `5` | beam search width. `1` = greedy: ~4-5x faster, a little less accurate |
| `NPUSUBS_DIARIZE_PARALLEL` | `0` | `1` runs the diarization of a request in parallel with its own transcription (faster, higher peak memory) |
| `NPUSUBS_VAD` | `1` | skip silence |
| `NPUSUBS_MUSIC` | `1` | keep and transcribe sustained loud non-speech (songs) |
| `NPUSUBS_DIARIZE` | `0` | default for `diarize=` (needs the setup below) |
| `NPUSUBS_SPEAKER_STYLE` | `dash` | `none` / `dash` / `label` |
| `NPUSUBS_MAX_LINE` | `42` | display columns per line (CJK counts double) |
| `NPUSUBS_GAP_SPLIT` | `0.5` | seconds of silence that start a new cue |
| `NPUSUBS_BIND` / `NPUSUBS_PORT` | `127.0.0.1` / `9091` | compose only. **There is no authentication**: keep it on a trusted network |

### Speaker diarization (optional)

Two engines, chosen with `diarization.engine` (or `diarization_engine=` per request, or in the bridge settings):

| Engine | Needs | Quality | Cost |
|---|---|---|---|
| `pyannote` | Hugging Face token and accepting the gated terms (below) | best: overlapped speech, fast exchanges | about 0.8x the audio length on 4 cores |
| `embedding` | **nothing** (a 27 MB public model is downloaded once, pinned and checksum-verified) | coarser: windows of 1.5 s, no overlap detection; the number of speakers depends on `diarization.embedding.threshold` | about 15x cheaper |
| `auto` (default) | | pyannote when it is usable, otherwise embedding | |

In a test on 4 minutes of dialogue the token-free engine found the same 3 speakers as pyannote and agreed with it on 88 %
of the speech time, in 16 s instead of several minutes; on a 30 s single-speaker clip it split one voice into two at
the default threshold, so give it `min_speakers`/`max_speakers` when you know them. It builds on the WeSpeaker ResNet34
embedding (the network pyannote uses) in the form published as `spurtcarl/finesub-speaker-embedding` (CC BY 4.0; credit
WeSpeaker and pyannote).

#### pyannote engine

Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio). The model is **gated**, so this engine needs a **free
Hugging Face account** that has accepted the model's licence terms once
(<https://huggingface.co/pyannote/speaker-diarization-community-1>) and a read token; the step-by-step is in
[docs/huggingface.md](docs/huggingface.md). Put the token in `config/secrets.env` as `HF_TOKEN`, then set
`NPUSUBS_DIARIZE=1` (or send `diarize=true` per request). It runs on the CPU, after the transcription of the same request.
**Never commit your token.** No account? Use the `embedding` engine.

## Measured (Ryzen 7 8845HS, 16 threads, large-v3-turbo)

* 4 minutes of Japanese anime audio (including an 80 s opening song), on a 4-core container:
  greedy decoding **~89 s**, beam search (default, width 5) **~406 s**. Beam search finds more of the dialogue
  (higher mean log-probability, -0.30 vs -0.36, and lines the greedy pass drops entirely). Encoder ~4 s per 30 s
  window on the NPU; the CPU decoder dominates the rest.
* Speaker diarization on the CPU (pyannote community-1) costs roughly 0.8x the audio length on 4 cores.
  Beam search plus diarization on the same 4 minutes: **~660 s** in total (about 2.7x the audio length),
  so a 24 minute episode takes about an hour. Plan for it as a background job and raise your client timeouts
  (Bazarr's Whisper provider, any bridge in front of it). Requests are processed one at a time.
* Running diarization in parallel with transcription (`NPUSUBS_DIARIZE_PARALLEL=1`) crashed the service once on a
  small container (4 cores), so it is off by default.
* Next to a running 30B LLM on the iGPU: the LLM keeps **~90 %** of its throughput while Whisper runs on the NPU,
  versus **~61 %** when Whisper runs on the iGPU too (Vulkan whisper.cpp), at the price of Whisper being ~1.7x slower.
  Measured with a real translation workload; baselines vary +-10 %.
* The NPU path is **slower** than a GPU. It is worth it when the GPU is busy with something else.

## Limitations

* Only the encoder is on the NPU (bf16); the decoder is the stock PyTorch CPU decoder.
* **One request at a time**, start to finish (decoding, NPU, CPU decoder, speaker diarization): the others wait in line, first
  come first served (`max_concurrent_jobs`, default 1; `/health` shows `jobs.running` and `jobs.queued`). Two at once slowed each
  other down by more than 10x in a test and can run a small container out of memory.
* No vocal separation yet (see roadmap). Dialogue over loud music can still degrade.
* Models and results depend on the upstream whisper-xdna kernels; large changes there may need an update here.

## Roadmap

* Optional vocal separation (e.g. [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp), MIT) in front of the
  music regions, to improve lyrics.
* `/inference` (whisper.cpp) endpoint.
* Smaller encoder contexts for short clips.

## Development

```bash
pip install numpy scipy fastapi httpx python-multipart
python -m unittest discover -s tests -v          # no NPU, model or torch needed
python subcfg/gen_schemas.py config.example/schemas   # regenerate the settings schemas (one table is the source of truth)
```

The unit tests also run in CI (`.github/workflows/tests.yml`). See [CHANGELOG.md](CHANGELOG.md) and
[SECURITY.md](SECURITY.md).

## Credits and licences

* NPU encoder: [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna) (MIT), built on
  [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) (Apache-2.0 with LLVM exception).
* [openai-whisper](https://github.com/openai/whisper) (MIT), [Silero VAD](https://github.com/snakers4/silero-vad) (MIT),
  [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT; the gated models have their own terms).
* Ideas (no code copied) from projects such as GeiserX/whisper-subs (GPL-3.0), stayallive/whisper-subtitles and
  giakinh0823/whisper-subtitle-generator.

This project: MIT, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
