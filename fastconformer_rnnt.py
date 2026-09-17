"""
Инференс модели NVIDIA NeMo FastConformer Hybrid RNNT в формате ONNX.

В отличие от CTC-модели (см. gigaam_ctc.py), здесь используется архитектура
encoder + decoder-joint с пошаговым (streaming-style) greedy-декодированием
и учётом скрытого состояния декодера между шагами.
"""

import re
from typing import List, Tuple

import librosa
import numpy as np
import onnxruntime as rt
from scipy.signal import get_window


class NemoPreprocessor:
    """Препроцессинг аудио: preemphasis -> STFT -> мел-спектрограмма -> нормализация."""

    def __init__(self, n_mels: int = 80):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.preemph = 0.97
        self.n_mels = n_mels
        self.log_zero_guard_value = float(2 ** -24)
        self.pad_to = 16
        self.normalize = "per_feature"

        # Параметры идентичны используемым в NeMo
        self.melscale_fbanks = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2,
            norm="slaney",
            htk=False,
        ).T.astype(np.float32)

        self.hann_window = get_window("hann", self.win_length, fftbins=True).astype(np.float32)

    def _preemphasis(self, waveforms: np.ndarray) -> np.ndarray:
        if self.preemph == 0.0:
            return waveforms
        preemphasized = np.zeros_like(waveforms)
        preemphasized[:, 0] = waveforms[:, 0]
        preemphasized[:, 1:] = waveforms[:, 1:] - self.preemph * waveforms[:, :-1]
        return preemphasized

    def _stft(self, waveforms: np.ndarray) -> np.ndarray:
        """Ручная реализация STFT на NumPy (без сторонних DSP-библиотек)."""
        batch_size, signal_length = waveforms.shape
        stft_results = []

        for b in range(batch_size):
            signal = waveforms[b]

            pad_width = self.n_fft // 2
            signal_padded = np.pad(signal, pad_width, mode="reflect")

            if self.win_length != self.n_fft:
                window = np.zeros(self.n_fft, dtype=np.float32)
                start_idx = (self.n_fft - self.win_length) // 2
                window[start_idx:start_idx + self.win_length] = self.hann_window
            else:
                window = self.hann_window.copy()

            n_frames = 1 + (len(signal_padded) - self.n_fft) // self.hop_length
            stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex64)

            for frame_idx in range(n_frames):
                start = frame_idx * self.hop_length
                end = start + self.n_fft

                if end <= len(signal_padded):
                    frame = signal_padded[start:end]
                    windowed_frame = frame * window
                    fft_result = np.fft.fft(windowed_frame, n=self.n_fft)
                    stft_matrix[:, frame_idx] = fft_result[: self.n_fft // 2 + 1]

            magnitude_squared = np.abs(stft_matrix) ** 2
            stft_results.append(magnitude_squared)

        return np.stack(stft_results, axis=0)

    def _compute_mel_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        # spectrogram: [batch, freq_bins, time]
        mel_spectrogram = np.matmul(spectrogram.transpose(0, 2, 1), self.melscale_fbanks)
        return mel_spectrogram.transpose(0, 2, 1)

    def _log_mel_spectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        return np.log(mel_spectrogram + self.log_zero_guard_value)

    def _normalize(self, features: np.ndarray, features_lens: np.ndarray) -> np.ndarray:
        if self.normalize != "per_feature":
            return features

        batch_size, n_mels, max_frames = features.shape
        normalized_features = np.zeros_like(features)

        for b in range(batch_size):
            seq_len = min(int(features_lens[b]), max_frames)
            if seq_len > 0:
                sequence = features[b, :, :seq_len]
                mean = np.mean(sequence, axis=1, keepdims=True)
                std = np.std(sequence, axis=1, keepdims=True, ddof=0)
                normalized_sequence = (sequence - mean) / (std + 1e-5)
                normalized_features[b, :, :seq_len] = normalized_sequence

        return normalized_features

    def get_seq_len(self, seq_len: np.ndarray) -> np.ndarray:
        return np.ceil(seq_len.astype(np.float32) / self.hop_length).astype(np.int64)

    def get_features(
        self, waveforms: np.ndarray, waveforms_lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        waveforms = self._preemphasis(waveforms)
        spectrogram = self._stft(waveforms)
        mel_spectrogram = self._compute_mel_spectrogram(spectrogram)
        log_mel_spectrogram = self._log_mel_spectrogram(mel_spectrogram)
        features_lens = self.get_seq_len(waveforms_lens)
        normalized_features = self._normalize(log_mel_spectrogram, features_lens)
        return normalized_features, features_lens


class OnnxConformerRNNT:
    """Инференс encoder + decoder-joint графов RNN-T через ONNX Runtime."""

    def __init__(self, model_files: dict):
        self._encoder = rt.InferenceSession(model_files["encoder"], providers=["CPUExecutionProvider"])
        self._decoder_joint = rt.InferenceSession(
            model_files["decoder_joint"], providers=["CPUExecutionProvider"]
        )
        self.vocab = self._load_vocab(model_files["vocab"])
        self._setup_token_indices()

    def _load_vocab(self, vocab_path: str) -> List[str]:
        vocab = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    token, idx = parts
                    idx = int(idx)
                    while len(vocab) <= idx:
                        vocab.append("")
                    vocab[idx] = token
                elif len(parts) == 1:
                    vocab.append(parts[0])
        while len(vocab) < 2561:
            vocab.append("<pad>")
        return vocab

    def _setup_token_indices(self):
        self._blank_idx = self.vocab.index("<blk>") if "<blk>" in self.vocab else 2560
        self._unk_idx = self.vocab.index("<unk>") if "<unk>" in self.vocab else 2304
        self._pad_idx = self.vocab.index("<pad>") if "<pad>" in self.vocab else 2560
        self._max_vocab_idx = len(self.vocab) - 1
        self._tokens_to_filter = {self._blank_idx, self._unk_idx, self._pad_idx}
        for i in range(self._max_vocab_idx + 1, len(self.vocab)):
            self._tokens_to_filter.add(i)

    def _encode(self, features: np.ndarray, features_lens: np.ndarray):
        encoder_out, encoder_out_lens = self._encoder.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": features, "length": features_lens},
        )
        return encoder_out, encoder_out_lens

    def _decode(self, prev_tokens: List[int], prev_state, encoder_out: np.ndarray):
        outputs, state1, state2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": encoder_out.astype(np.float32),
                "targets": np.array(
                    [[self._blank_idx if not prev_tokens else prev_tokens[-1]]], dtype=np.int32
                ),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": prev_state[0],
                "input_states_2": prev_state[1],
            },
        )
        return np.squeeze(outputs), -1, (state1, state2)

    def greedy_search(self, encoder_out: np.ndarray, encoder_out_len: np.ndarray) -> str:
        """Пошаговое greedy-декодирование с учётом состояния decoder-joint сети."""
        max_len = encoder_out.shape[2]
        state = (np.zeros((1, 1, 640), dtype=np.float32), np.zeros((1, 1, 640), dtype=np.float32))
        hyp = []

        for t in range(max_len):
            current_encoder_out = encoder_out[:, :, t : t + 1]
            logits, _, state = self._decode(hyp, state, current_encoder_out)
            logits = logits.copy()

            logits = logits - np.max(logits)
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-12)
            next_token = np.argmax(probs).item()

            # Порог уверенности: низкая вероятность трактуется как blank
            if probs[next_token] < 0.4:
                next_token = self._blank_idx

            if next_token != self._blank_idx:
                hyp.append(next_token)

        return self._postprocess(hyp)

    def _postprocess(self, decoded_ids: List[int]) -> str:
        valid_tokens = [
            self.vocab[tok_id]
            for tok_id in decoded_ids
            if tok_id not in self._tokens_to_filter and tok_id <= self._max_vocab_idx
        ]
        text = "".join(valid_tokens).replace("▁", " ").strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1", text)

        # Убираем идущие подряд дубли слов (артефакт greedy-декодирования)
        words = text.split()
        cleaned_words = []
        last_word = None
        for word in words:
            if word and (word.lower() != (last_word or "").lower() or len(word) <= 2):
                cleaned_words.append(word)
                last_word = word

        return " ".join(cleaned_words).strip()


if __name__ == "__main__":
    audio, sr = librosa.load("was_ist.wav", sr=16000)
    audio_batch = audio.reshape(1, -1)
    audio_len = np.array([audio.shape[0]], dtype=np.int64)

    preprocessor = NemoPreprocessor(n_mels=80)
    features, features_len = preprocessor.get_features(audio_batch, audio_len)

    model_files = {
        "encoder": "encoder-model.onnx",
        "decoder_joint": "decoder_joint-model.onnx",
        "vocab": "vocab.txt",
    }
    onnx_model = OnnxConformerRNNT(model_files)

    encoder_out, encoder_out_len = onnx_model._encode(
        features.astype(np.float32), features_len.astype(np.int64)
    )
    transcription = onnx_model.greedy_search(encoder_out, encoder_out_len)

    print("Transcription:", transcription)
