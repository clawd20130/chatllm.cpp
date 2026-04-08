# OmniVoice Notes

This note records the current `chatllm.cpp` OmniVoice integration status, the architectural choices behind it, and the main lessons from bringing OmniVoice into the native runtime.

## Status

There are now two OmniVoice paths:

* Bridge mode: `-m :omnivoice`
  * Delegates requests to a running local OmniVoice worker-sdk service.
  * Useful as a reference implementation and fallback.
* Native mode: `-m /path/to/omnivoice-native.bin`
  * Runs OmniVoice generation inside `chatllm.cpp`.
  * Supports auto voice, voice design, and an approximate native clone path.

## Conversion Notes

The native path depends on more than the Qwen3 backbone and the decode-side DAC stack.

The converter must include:

* `llm.*` -> `model.*`
* `audio_embeddings.*` -> `omnivoice.audio_embeddings.*`
* `audio_heads.*` -> `omnivoice.audio_heads.*`
* `acoustic_encoder.*` -> `audio_tokenizer.acoustic_encoder.*`
* `acoustic_decoder.*` -> `audio_tokenizer.acoustic_decoder.*`
* `fc.*` -> `audio_tokenizer.fc.*`
* `fc2.*` -> `audio_tokenizer.fc2.*`
* `quantizer.quantizers.*.project_in.*` -> `audio_tokenizer.quantizer.quantizers.*.project_in.*`
* `quantizer.quantizers.*.project_out.*` -> `audio_tokenizer.quantizer.quantizers.*.project_out.*`
* `quantizer.quantizers.*.codebook.embed` -> `audio_tokenizer.quantizer.quantizers.*.codebook.weight`

OmniVoice also needs tokenizer metadata from `audio_tokenizer/config.json` and `preprocessor_config.json`, especially:

* `sample_rate`
* `hop_length`
* `frame_rate`
* `codebook_size`
* `codebook_dim`
* `num_quantizers`
* DAC `downsampling_ratios` and `upsampling_ratios`
* semantic hidden size

## Native Runtime Design

The native implementation is split into two pieces:

* OmniVoice generator
  * Qwen3 backbone with non-causal attention.
  * Audio logits head over `num_audio_codebook * audio_vocab_size`.
  * Iterative masked decoding with CFG.
* Higgs tokenizer runtime
  * Decode path: RVQ decode -> `fc2` -> DAC decoder.
  * Encode path: DAC acoustic encoder -> `fc` -> RVQ encode.

The current native clone path uses the acoustic encoder and RVQ encode path only. The HuBERT semantic branch is not native yet.

That means:

* native auto voice: implemented
* native voice design: implemented
* native voice clone: implemented, but approximate
* parity with Python OmniVoice clone quality: not reached yet

## Key Lessons

### 1. Decode-only was not enough

Initial native support could synthesize audio from generated tokens, but could not clone because reference audio tokens still depended on the Python tokenizer encode path.

The minimum native clone path required:

* acoustic encoder
* tokenizer `fc`
* RVQ `project_in`
* codebook lookup
* residual subtraction across quantizers

### 2. The HuBERT semantic branch is the real blocker for full parity

OmniVoice clone quality depends on the Higgs semantic branch as well as the acoustic branch.

`chatllm.cpp` does not currently have a native HuBERT implementation that can be dropped into this tokenizer.

As a result, the current clone path zero-fills the semantic half before `fc`, which is enough to produce usable reference audio tokens, but it is still an approximation.

### 3. Hidden states vs. logits mattered

One early native bug came from feeding text vocab logits into `audio_heads`.

The correct input is `transformer->last_hidden_state`, not the text LM head output.

### 4. DAC tensor layout was easy to get wrong

Before the DAC decoder, the post-`fc2` tensor had to be permuted into channel-first layout. Without that, convolution shapes looked superficially valid but decoding failed.

### 5. `ConvTranspose1d.output_padding` support was required

OmniVoice Higgs DAC uses odd upsampling ratios such as `5` and `3`, which imply `output_padding = 1` in some decoder layers.

`chatllm.cpp` transposed-convolution code had to be extended to handle that case.

### 6. Quantizer cache extraction needed an `F32` fast path

The native clone encoder prepares CPU-side caches for `project_in`, `project_out`, and codebook weights.

Using the generic quantized-tensor conversion path for already-`F32` tensors caused a crash during cache preparation. `F32` tensors must be read directly.

### 7. `ref_text` must stay mandatory

The native path does not auto-transcribe reference audio.

If `ref_audio_file` is provided without `ref_text`, fail early and clearly. Do not silently invent an ASR fallback.

## Output Format

`--tts_export` currently writes raw 24kHz mono `pcm_s16le`, not a WAV container.

For inspection with normal audio tools, wrap it first:

```sh
ffmpeg -y -f s16le -ar 24000 -ac 1 -i /tmp/out.pcm /tmp/out.wav
```

One benchmarking caveat:

* `main --tts_export ...` still calls `ffplay` after synthesis when `ffplay` is available.
* End-to-end CLI wall time therefore includes audio playback time unless playback is disabled.
* `--no_play` now disables `ffplay` without stubbing binaries out.
* For pure generation benchmarks, keep the model resident in one process and use `--no_play`.

## Vulkan Benchmark Notes

Measured on this machine with:

* backend: Vulkan
* GPU: AMD Radeon 780M Graphics (`RADV PHOENIX`)
* model: native OmniVoice (`/tmp/omnivoice-native.bin`)
* prompt: `这是一次在 AMD 780M 集成显卡上通过 Vulkan 运行原生 OmniVoice 自动音色语音合成的性能测试。`
* generation: auto voice, `language=Chinese`, `num_step=32`
* output: 24kHz mono PCM, `8.6s`

Observed numbers:

* end-to-end CLI timing after warmup, including process startup and post-generation playback path: about `20.8s` wall time for `8.6s` audio, or `RTF ~= 2.42`
* earlier pure generation timing with one resident interactive process: about `12.3s` wall time for `8.6s` audio, or `RTF ~= 1.43`
* current stable pure generation timing after adding `--no_play` and a CPU scoring fast path:
  * setup: one resident interactive process, `2` warmup runs, then `5` measured runs
  * average: `10.98s` wall time for `8.6s` audio, or `RTF ~= 1.28`
  * stable tail (`runs 3-5`): `11.05s` wall time for `8.6s` audio, or `RTF ~= 1.28`

Interpretation:

* on this 780M Vulkan path, native OmniVoice auto TTS is not yet real-time at `num_step=32`
* stable pure generation speed is now about `0.78x` real-time
* relative to the previous `RTF ~= 1.43` baseline, the current stable path is about `10.8%` faster
* end-to-end shell timing overstates model cost if audio playback is not excluded

One optimization experiment did not survive:

* batching CFG conditional and unconditional branches with per-batch attention masks matched the Python reference structure
* on this current Vulkan 780M path, that version regressed wall time, so it was not kept as the default runtime path

## Validation Commands

Convert:

```sh
/home/kevinzhow/github/omnivoice-worker-sdk/.venv/bin/python ./convert.py \
  -i /path/to/OmniVoice \
  -a OmniVoice \
  -o /tmp/omnivoice-native.bin \
  -n OmniVoice \
  -t f16
```

Auto voice:

```sh
./build/bin/main -m /tmp/omnivoice-native.bin \
  -p "Hello from native OmniVoice." \
  --tts_export /tmp/omnivoice-native-auto.pcm \
  --set language English \
  --set num_step 4
```

Voice design:

```sh
./build/bin/main -m /tmp/omnivoice-native.bin \
  -p "This is native OmniVoice voice design." \
  --tts_export /tmp/omnivoice-native-design.pcm \
  --set language English \
  --set instruct "female, low pitch" \
  --set num_step 4
```

Voice clone:

```sh
./build/bin/main -m /tmp/omnivoice-native.bin \
  -p "This sentence should reuse the reference voice." \
  --tts_export /tmp/omnivoice-native-clone.pcm \
  --set language English \
  --set ref_audio_file /path/to/reference.wav \
  --set ref_text "This is the reference transcript." \
  --set num_step 4
```

## Next Steps

The next quality milestone is native HuBERT semantic feature extraction for reference audio.

After that:

* revisit clone parity against the Python runtime
* add long-text chunking for native OmniVoice
* consider WAV output in `--tts_export` instead of raw PCM
