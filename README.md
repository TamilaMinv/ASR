# ASR ONNX Inference: Conformer CTC and FastConformer RNN-T

Implementation of automatic speech recognition (ASR) inference pipelines for neural network models exported to the ONNX format. The project was completed as part of an undergraduate training practice at the Department of Applied Mathematics, Novosibirsk State Technical University (NSTU).

The main objective was to study how modern ASR systems work beyond high-level inference APIs by implementing the main stages of the speech-recognition pipeline: audio preprocessing, ONNX inference, and greedy decoding.

## Project Objectives

- Study the architecture and inference requirements of modern ASR systems.
- Implement audio preprocessing and mel-spectrogram generation.
- Implement ONNX inference and CTC decoding for **GigaAM Conformer v2**.
- Implement preprocessing, ONNX inference, and RNN-T decoding for **NVIDIA NeMo FastConformer Hybrid**.
- Compare the reproduced inference outputs with the expected outputs of the original models.

## Models

| Model | Architecture | Decoding approach |
|---|---|---|
| GigaAM Conformer v2 | Conformer | CTC (Connectionist Temporal Classification) |
| NVIDIA NeMo FastConformer Hybrid | FastConformer with encoder, prediction network, and joint network | RNN-T (Recurrent Neural Network Transducer) |

The two models represent different approaches to sequence decoding. CTC produces frame-level token predictions and removes repeated tokens and the blank symbol during post-processing. RNN-T performs sequential prediction while incorporating the history of previously predicted tokens.

## Inference Pipeline

The implemented pipeline follows the main stages below:

```text
WAV audio
   ↓
Audio loading and normalization
   ↓
STFT
   ↓
Mel-spectrogram
   ↓
Logarithmic transformation / feature normalization
   ↓
ONNX Runtime inference
   ↓
Greedy decoding (CTC or RNN-T)
   ↓
Transcribed text
```

### Audio preprocessing

For the GigaAM pipeline, the project implements the core signal-processing operations explicitly, including:

- audio loading and conversion to a single channel when necessary;
- resampling to 16 kHz when required;
- Short-Time Fourier Transform (STFT);
- construction of a mel filter bank;
- mel-spectrogram computation;
- logarithmic transformation of the extracted features.

The NumPy-generated spectrogram and mel-spectrogram were compared with reference PyTorch outputs using `np.testing.assert_allclose`, providing a numerical check of the custom preprocessing implementation.

For the FastConformer pipeline, preprocessing follows the parameters required by the NeMo model, including pre-emphasis, STFT, an 80-bin mel filter bank, feature normalization, and input padding.

### CTC decoding

For **GigaAM Conformer v2**, greedy CTC decoding is performed by selecting the most probable token at each time step and then removing consecutive repetitions and the CTC blank symbol.

### RNN-T decoding

For **FastConformer Hybrid**, inference uses the encoder together with the decoder/joint network. Greedy RNN-T decoding predicts tokens step by step while updating the decoder state so that previous predictions contribute to subsequent decoding decisions.

## Validation

The reproduced inference pipeline was checked against expected model outputs on several audio samples.

| Model | Audio sample | Language | Expected output | Obtained output |
|---|---|---|---|---|
| GigaAM Conformer v2 (CTC) | `checking.wav` | Russian | `Всем привет это проверка связи.` | `Всем привет это проверка связи.` |
| FastConformer Hybrid (RNN-T) | `english.wav` | English | `Hello, what is your name?` | `Hello, what is your name?` |
| FastConformer Hybrid (RNN-T) | `was_ist.wav` | German | `Was ist deine Staatsbürgerschaft?` | `Was ist deine Staatsbürgerschaft?` |

For all reported test samples, the obtained transcription matched the expected transcription.

## Technologies

- **Python**
- **NumPy** — numerical processing and custom signal-processing operations
- **ONNX Runtime** — neural network inference
- **SoundFile** — audio loading in the GigaAM implementation
- **librosa** and **SciPy** — audio preprocessing utilities used in the FastConformer implementation
- **PyTorch / torchaudio** — used in the GigaAM development code for reference comparison and resampling when required

## Repository Structure

```text
├── GigaAM.py          # GigaAM Conformer v2 preprocessing, ONNX inference, and CTC decoding
├── FastConformer.py   # NeMo FastConformer Hybrid preprocessing, ONNX inference, and RNN-T decoding
└── README.md
```

> The ONNX model weights and test audio files are not included in the repository and must be provided separately.

## Model Files

The implementations require exported ONNX model files:

- **GigaAM Conformer v2:** `v2_ctc.onnx`
- **NeMo FastConformer Hybrid:** encoder and decoder/joint ONNX models, together with the corresponding vocabulary file

Model sources:

- [GigaAM](https://github.com/salute-developers/GigaAM)
- [NVIDIA NeMo Multilingual FastConformer Hybrid](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc)

## Running the Project

Install the dependencies required by the implementations:

```bash
pip install numpy onnxruntime soundfile librosa scipy torch torchaudio
```

Place the required ONNX model files and audio samples in the paths expected by the scripts, then run:

```bash
python GigaAM.py
python FastConformer.py
```

The file paths in the scripts may need to be adjusted for the local environment.

## Key Learning Outcomes

This project provided practical experience with the internal stages of modern speech-recognition systems rather than relying only on high-level ASR APIs. It involved implementing and validating audio feature extraction, working directly with ONNX model inputs and outputs, and implementing two different sequence-decoding approaches: CTC and RNN-T.

The project also demonstrated how pretrained ASR architectures can be executed through ONNX Runtime and how their preprocessing and decoding pipelines can be reproduced and examined at a lower level.
