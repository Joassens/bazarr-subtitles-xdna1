# Changes relative to the upstream bridge (Apache License 2.0, section 4)

Upstream: https://github.com/McCloudS/bazarr-openai-whisperbridge

`bridge.py` in this directory is a modified version of the upstream file. Main changes:

* Several `WHISPER_ASR_URL` backends in priority order, with failover and temporary skipping of offline ones;
  `WHISPER_PREFER_NATIVE`, configurable `WHISPER_ASR_TIMEOUT`.
* Native whisper-asr-webservice path with word timestamps and speaker labels; a subtitle is split on speaker change.
* Language-aware translation prompt (with Japanese rules), context lines from earlier translated cues,
  honorific handling, optional dedicated translation LLM.
* Accuracy passes: scene bible and conservative review; style-only edits are rejected.
* Filters: repetition loops, recurring hallucinated lines, phantom first cue; OP/ED filename skip.
* Diarization only unloads the LLM when whisper and the LLM share a host.

The upstream copyright and the Apache-2.0 licence (see `LICENSE` in this directory) apply to the parts taken from it.
New code in this fork is offered under the same licence for this directory.
