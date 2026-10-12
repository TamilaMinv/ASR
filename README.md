# ASR Inference with ONNX: GigaAM CTC and FastConformer RNN-T

Python implementations of automatic speech recognition (ASR) inference using exported ONNX models. Developed during an undergraduate training placement in Applied Mathematics and Informatics at **Novosibirsk State Technical University (NSTU)** in 2025.

Rather than relying on a high-level transcription API, this project implements audio feature extraction, ONNX Runtime inference, and token decoding for two different ASR architectures.

## Overview

| Implementation | Model | Decoder | Input features |
| --- | --- | --- | --- |
| [`gigaam_ctc.py`](gigaam_ctc.py) | GigaAM Conformer v2 | Greedy CTC | Custom NumPy log-mel spectrogram, 64 mel filters |
| [`fastconformer_rnnt.py`](fastconformer_rnnt.py) | NVIDIA NeMo FastConformer Hybrid | Simplified frame-wise RNN-T greedy decoder | Pre-emphasis, STFT, 80 mel filters, per-feature normalization |

Both scripts use **ONNX Runtime** to execute pretrained models; they do not train new ASR models.

## Processing pipeline

```text
Audio file
   |
Audio loading at 16 kHz
   |
STFT -> mel filter bank -> log-mel features
   |
ONNX Runtime inference
   |
CTC or RNN-T decoding
   |
Transcribed text
```

### GigaAM Conformer v2 (CTC)

The `SpeechRecognizer` class implements WAV loading with `soundfile`, a NumPy-based Hann-windowed STFT, triangular mel filterbank construction, log-mel features, and CTC greedy decoding. Repeated token predictions and the blank token are removed during decoding.

**Input requirement:** `gigaam_ctc.py` expects a **16 kHz WAV** file and uses the first audio channel when a file is multichannel. It **does not resample** audio automatically; another sampling rate raises a `ValueError`. The ONNX file must have the input/output layout expected by this implementation.

### NVIDIA NeMo FastConformer Hybrid (RNN-T)

The `NemoPreprocessor` class implements pre-emphasis, a NumPy-based STFT, a mel filterbank created with `librosa`, logarithmic scaling, and per-feature normalization. The `OnnxConformerRNNT` class runs separate encoder and decoder-joint ONNX sessions with decoder-state tracking.

**Decoder scope:** the implementation performs one token decision per encoder frame and uses a confidence threshold of `0.4` plus text post-processing. This is a simplified, project-specific greedy decoder, not a complete general-purpose RNN-T beam search or streaming service. `pad_to` is configured in the preprocessor but input padding is not implemented in the supplied script.

## Repository structure

```text
.
├── README.md
├── gigaam_ctc.py
└── fastconformer_rnnt.py
```

The model weights, vocabulary and example audio files are **not included** and must be obtained separately. The model file names and paths below match those hard-coded in the scripts.

## Requirements

- Python 3
- `numpy`
- `onnxruntime` (CPU execution is sufficient for the included scripts)
- `soundfile` (for GigaAM WAV loading)
- `librosa` and `scipy` (for the FastConformer implementation)

Install dependencies:

```bash
python -m pip install numpy onnxruntime soundfile librosa scipy
```

PyTorch/torchaudio are **not imported by these two scripts**. They were used in development/reference comparisons documented in the training report, but are not required for the supplied inference code.

## Run GigaAM CTC

Place the compatible ONNX model and WAV test file at the paths used in `gigaam_ctc.py`:

```text
.
├── gigaam_ctc.py
├── checking.wav
└── onnx/
    └── v2_ctc.onnx
```

Then run:

```bash
python gigaam_ctc.py
```

The script loads `onnx/v2_ctc.onnx`, transcribes `checking.wav`, and prints the transcription and elapsed time. The audio must already be sampled at 16 kHz. Change the paths under `if __name__ == "__main__":` if your files are elsewhere.

## Run FastConformer RNN-T

Provide the exported model files, vocabulary, and audio example:

```text
.
├── fastconformer_rnnt.py
├── was_ist.wav
├── encoder-model.onnx
├── decoder_joint-model.onnx
└── vocab.txt
```

Run:

```bash
python fastconformer_rnnt.py
```

The script loads `was_ist.wav` through `librosa` at 16 kHz, prepares the features, and runs the encoder and decoder-joint graphs on CPU. Its graph input/output names, decoder-state shapes and vocabulary indexing are tied to the particular ONNX exports used during development; other exports may require adaptation.

## Reported validation examples

The undergraduate practice report documents the following matches between expected and obtained transcriptions:

| Model | Example audio | Language | Expected transcription | Reported transcription |
| --- | --- | --- | --- | --- |
| GigaAM Conformer v2 (CTC) | `checking.wav` | Russian | Всем привет это проверка связи. | Всем привет это проверка связи. |
| NeMo FastConformer Hybrid (RNN-T) | `english.wav` | English | Hello, what is your name? | Hello, what is your name? |
| NeMo FastConformer Hybrid (RNN-T) | `was_ist.wav` | German | Was ist deine Staatsbürgerschaft? | Was ist deine Staatsbürgerschaft? |

These are the **three examples documented in the 2025 training report**, not a large-scale accuracy benchmark. The ONNX weights and audio files are not included here, so this repository by itself cannot reproduce those transcriptions without the corresponding external assets.

## Implementation notes

- **CTC:** decoding selects the highest-scoring token at each time step, removes adjacent repeats, and skips the blank token.
- **RNN-T:** decoding processes encoder frames sequentially, updates the decoder state, applies a confidence threshold, and performs heuristic text cleanup.
- **Preprocessing:** both scripts explicitly implement signal-processing steps, with different mel-filtering and normalization settings to match their respective model pipelines.
- **Model compatibility:** both implementations assume specific ONNX model interfaces and pre-existing pretrained weights.

## Model references

- [GigaAM — SberDevices / Salute Developers](https://github.com/salute-developers/GigaAM)
- [NVIDIA NeMo multilingual FastConformer Hybrid model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc)

## Project context

This project was completed as an undergraduate educational placement at NSTU. The work focused on understanding ASR model architectures and writing lower-level Python implementations of audio preprocessing and ONNX-based inference for CTC and RNN-T models.
