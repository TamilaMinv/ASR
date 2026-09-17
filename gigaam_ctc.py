"""
Инференс модели GigaAM Conformer v2 (CTC) в формате ONNX.

Реализует полный пайплайн: чтение WAV -> препроцессинг (STFT, мел-спектрограмма,
логарифмирование) -> инференс через ONNX Runtime -> CTC greedy decoding -> текст.

Единственные зависимости: numpy, onnxruntime, soundfile (torch/torchaudio
используются здесь только для ресемплинга и сверки времени выполнения —
для чистого инференса не требуются).
"""

import time

import numpy as np
import onnxruntime as ort
import soundfile as sf


class SpeechRecognizer:
    """Инференс ASR-модели GigaAM Conformer v2 (CTC) из ONNX-файла."""

    # Словарь для преобразования индексов в символы
    VOCAB = [
        " ", "а", "б", "в", "г", "д",
        "е", "ж", "з", "и", "й", "к",
        "л", "м", "н", "о", "п", "р",
        "с", "т", "у", "ф", "х", "ц",
        "ч", "ш", "щ", "ъ", "ы", "ь",
        "э", "ю", "я",
    ]

    def __init__(self, onnx_path, sample_rate=16000, n_mels=64):
        self.onnx_path = onnx_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = sample_rate // 40
        self.hop_length = sample_rate // 100
        self.blank_id = 33

        self.session = ort.InferenceSession(onnx_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        # Предварительно вычисляем и кэшируем мел-фильтры
        self.mel_filter_bank = self.create_mel_filterbank(self.n_fft)

    def id_to_char(self, idx):
        if 0 <= idx < len(self.VOCAB):
            return self.VOCAB[idx]
        return ""

    def ctc_decode(self, pred_ids):
        """CTC greedy decoding: убирает повторы и blank-токены."""
        prev = None
        result = []
        for p in pred_ids[0]:
            if p != prev and p != self.blank_id:
                result.append(p)
            prev = p
        return "".join(self.id_to_char(i) for i in result)

    def decode_outputs(self, outputs):
        logits = outputs[0]  # [batch, seq_len, vocab_size]
        pred_ids = np.argmax(logits, axis=-1)  # [batch, seq_len]
        return self.ctc_decode(pred_ids)

    @staticmethod
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def create_mel_filterbank(self, n_fft):
        """Строит треугольный банк мел-фильтров (аналог torchaudio.MelScale)."""
        low_freq_mel = self.hz_to_mel(0)
        high_freq_mel = self.hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)

        hz_points = self.mel_to_hz(mel_points)
        fft_bins = np.floor((n_fft + 1) * hz_points / self.sample_rate)

        filters = np.zeros((self.n_mels, n_fft // 2 + 1))
        for i in range(self.n_mels):
            for j in range(int(fft_bins[i]), int(fft_bins[i + 1])):
                filters[i, j] = (j - fft_bins[i]) / (fft_bins[i + 1] - fft_bins[i])
            for j in range(int(fft_bins[i + 1]), int(fft_bins[i + 2])):
                filters[i, j] = (fft_bins[i + 2] - j) / (fft_bins[i + 2] - fft_bins[i + 1])

        # Нормализация: сумма каждого фильтра равна 1
        filter_sums = np.sum(filters, axis=1, keepdims=True)
        filter_sums = np.where(filter_sums == 0, 1, filter_sums)
        filters = filters / filter_sums

        return filters

    def load_wav(self, wav_path):
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        waveform = waveform[:, 0]  # берём первый канал, если стерео

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Ожидается частота дискретизации {self.sample_rate} Гц, "
                f"получено {sample_rate} Гц. Выполните ресемплинг заранее."
            )

        return waveform

    def compute_spectrogram(self, waveform, n_fft, hop_length):
        """STFT с окном Ханна, реализованное на чистом NumPy."""
        window = np.hanning(n_fft)

        pad_length = n_fft // 2
        padded_waveform = np.pad(waveform, (pad_length, pad_length), mode="reflect")

        n_frames = 1 + (len(padded_waveform) - n_fft) // hop_length
        frames = np.zeros((n_frames, n_fft))

        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            frames[i] = padded_waveform[start:end] * window

        spectrum = np.fft.rfft(frames)
        spectrogram = np.abs(spectrum) ** 2
        return spectrogram.T

    def extract_features(self, waveform):
        spectrogram = self.compute_spectrogram(waveform, self.n_fft, self.hop_length)
        mel_spectrogram = np.dot(self.mel_filter_bank, spectrogram)
        features = np.log(np.clip(mel_spectrogram, 1e-9, 1e9))
        return features

    def run_inference(self, features):
        features = features[np.newaxis, :, :]  # (1, n_mels, seq_len)
        feature_lengths = np.array([features.shape[2]], dtype=np.int64)
        inputs = {
            self.input_names[0]: features.astype(np.float32),
            self.input_names[1]: feature_lengths,
        }
        return self.session.run(None, inputs)

    def transcribe(self, wav_path):
        waveform = self.load_wav(wav_path)
        features = self.extract_features(waveform)
        outputs = self.run_inference(features)
        return self.decode_outputs(outputs)


if __name__ == "__main__":
    onnx_model_path = "onnx/v2_ctc.onnx"
    wav_path = "checking.wav"

    recognizer = SpeechRecognizer(onnx_model_path)

    start_time = time.time()
    transcription = recognizer.transcribe(wav_path)
    recognition_time = time.time() - start_time

    print("Распознанный текст:", transcription)
    print(f"Время распознавания: {recognition_time:.4f} секунд")
