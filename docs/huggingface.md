# Hugging Face account (only for the best speaker recognition)

**You do not need this to use the project.** Transcription, translation and the web UI work without any account. A free
Hugging Face account is needed for one optional feature only: the `pyannote` speaker-diarization engine (the best
quality: overlapping speech, fast exchanges). The model behind it is *gated*: its author wants every user to accept its
licence terms first, and Hugging Face only lets an account that did so download it.

| Engine | Account needed? |
|---|---|
| `embedding` | No. A small public model is downloaded once (pinned and checksum-verified). Coarser, about 15x cheaper. |
| `pyannote` | **Yes**: free account, licence accepted, access token. |
| `auto` (default) | Uses `pyannote` when the token works, otherwise `embedding`. So without an account you still get speaker labels. |

## One-time setup (5 minutes)

1. **Create an account** (free) at <https://huggingface.co/join>.
2. **Accept the licence terms.** Open <https://huggingface.co/pyannote/speaker-diarization-community-1> while signed in,
   read the conditions and press the button to accept them. This is *you* agreeing to the model author's terms; nobody can do
   it for you, and it is tied to your account.
3. **Create an access token**: <https://huggingface.co/settings/tokens> → *Create new token* → type **Read** is enough.
   Copy it (it starts with `hf_`).
4. **Give the token to the stack**: in your config folder, copy `secrets.env.example` to `secrets.env`, `chmod 600` it and
   set `HF_TOKEN=hf_...`. Restart npusubs once (`docker compose restart npusubs`).
5. **Check**: `curl localhost:9091/health` shows the diarization status. `ready` means it works; `no HF_TOKEN` or
   `model unavailable: ... accept the terms` means step 2 or 4 is not done yet.

Then switch it on: setting `diarization.engine` (web UI → Configuration), or `diarize=true` per request. Without it nothing
speaker-related happens.

## Good to know

* The first use downloads the models (a few hundred MB) into the `hf` volume; after that it runs offline.
* The token is a secret: it only belongs in `secrets.env` (never in the JSON files, the compose file or git). The web UI never shows it.
* If you do not want an account at all, set `diarization.engine` to `embedding`.
* Terms can change at the model author's side; if the status flips to "model unavailable", open the model page again and
  check whether you have to accept new conditions.
