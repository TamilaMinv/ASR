# ASR ONNX Inference

Python-пакет для инференса моделей автоматического распознавания речи (ASR), экспортированных в формат ONNX, без использования тяжёлых ML-фреймворков (PyTorch, TensorFlow) на этапе исполнения.

Выполнено в рамках учебной практики, НГТУ, кафедра прикладной математики.

## Идея проекта

Изучить архитектуры современных ASR-систем «изнутри» — не через готовый high-level API, а реализовав препроцессинг аудио, инференс и декодирование самостоятельно, на минимальном наборе зависимостей: **numpy + onnxruntime**.

Поддержаны две принципиально разные архитектуры декодирования:

| Модель | Архитектура | Подход к декодированию |
|---|---|---|
| GigaAM Conformer v2 | Conformer (conv + self-attention) | CTC |
| NeMo FastConformer Hybrid | FastConformer, encoder + decoder-joint | RNN-Transducer |

## Как это работает

**Пайплайн (общий для обеих моделей):**

```
WAV-файл → нормализация → STFT → мел-спектрограмма → логарифмирование
        → инференс ONNX-модели → декодирование логитов → текст
```

- **STFT и мел-фильтры реализованы вручную на NumPy** (без librosa/torchaudio для GigaAM-пайплайна) — это позволило детально разобраться в математике препроцессинга и добиться числового совпадения с эталонной PyTorch-реализацией (проверено через `np.testing.assert_allclose`).
- **CTC-декодирование** (`gigaam_ctc.py`): на каждом шаге выбирается токен с максимальной вероятностью, затем убираются повторы и blank-символы.
- **RNN-T декодирование** (`fastconformer_rnnt.py`): пошаговое предсказание с сохранением скрытого состояния decoder-joint сети между шагами — модель "помнит" контекст предыдущих предсказаний, что ближе к тому, как RNN-T используется в потоковом (streaming) распознавании.

## Результаты

Корректность инференса проверялась сравнением с эталонным выводом моделей:

| Модель | Файл | Ожидалось | Получено |
|---|---|---|---|
| GigaAM Conformer v2 (CTC) | checking.wav (RU) | «Всем привет это проверка связи.» | «Всем привет это проверка связи.» ✅ |
| FastConformer Hybrid RNNT | english.wav (EN) | «Hello, what is your name?» | «Hello, what is your name?» ✅ |
| FastConformer Hybrid RNNT | was_ist.wav (DE) | «Was ist deine Staatsbürgerschaft?» | «Was ist deine Staatsbürgerschaft?» ✅ |

FastConformer Hybrid — мультиязычная модель, поэтому тестировалась на трёх языках (RU/EN/DE) для проверки устойчивости пайплайна.

## Стек

Python, NumPy, ONNX Runtime, soundfile / librosa (только для чтения аудио)

## Структура репозитория

```
├── gigaam_ctc.py            # инференс GigaAM Conformer v2 (CTC)
├── fastconformer_rnnt.py    # инференс NeMo FastConformer Hybrid (RNN-T)
└── README.md
```

## Запуск

```bash
pip install numpy onnxruntime soundfile librosa scipy
python gigaam_ctc.py
python fastconformer_rnnt.py
```

Требуются ONNX-веса моделей (не включены в репозиторий из-за размера):
- [GigaAM](https://github.com/salute-developers/GigaAM) — `v2_ctc.onnx`
- [NVIDIA NeMo FastConformer Hybrid](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_multilingual_fastconformer_hybrid_large_pc) — `encoder-model.onnx`, `decoder_joint-model.onnx`, `vocab.txt`

## Вывод

Проект показал, что развёртывание сложных ASR-архитектур (Conformer, RNN-T) возможно без тяжёлых ML-фреймворков — только на ONNX Runtime и NumPy. Такой подход снижает вес зависимостей приложения и упрощает деплой на CPU-only среды, что актуально для встраивания распознавания речи в лёгкие клиентские приложения или serverless-окружения.
