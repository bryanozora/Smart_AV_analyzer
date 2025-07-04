# 🎧 Smart AV Analyzer – Smart Audio Transcriber, Diarizer, Summarizer, and Question Answering

**Smart AV Analyzer** is an AI-powered pipeline that transcribes audio, performs speaker diarization, labels speaker turns, generates conversation summaries, and answers questions based on the conversation. It also includes fine-tuning of the `FLAN-T5` language model for the task of dialogue summarization. This project was developed as a final assignment for the **Deep Learning** courses at **Petra Christian University**.

---

## 📚 Academic Context

This project demonstrates a real-world implementation of generative NLP on audio data using state-of-the-art models for automatic speech recognition (ASR), speaker diarization, text summarization, and question answering. It also showcases how large language models can be fine-tuned for better summarization quality on domain-specific dialogues.

---

## ✨ Features

### 🔊 Audio Transcription Pipeline
- **Automatic Speech Recognition** using OpenAI’s `whisper-base` model.
- **Speaker Diarization** using `pyannote.audio` to detect speaker boundaries.
- **Speaker Tagging**: Merge speaker information with ASR results.
- **Subtitle Generation**: Create `.srt` subtitle files with timestamps and speaker labels.
- **Text Summarization**: Summarize the full transcript using `google/flan-t5-base`.
- **Question Answering**: Answer Questions using deepset/roberta-base-squad2

### 🧠 FLAN-T5 Fine-Tuning Pipeline
- **Dialogsum Dataset**: Dialogsum dataset obtained from hugginface https://huggingface.co/datasets/knkarthick/dialogsum.
- **Model Training**: Fine-tune `google/flan-t5-base` on the previously mentioned dataset.
- **Evaluation**: Displaying metrics like accuracy, recall, loss, etc.

---

## 🛠️ Technologies & Models

- **Programming Language**: Python
- **Models**:
  - [`openai/whisper-base`](https://huggingface.co/openai/whisper-base) – for audio transcription
  - [`pyannote/speaker-diarization`](https://huggingface.co/pyannote/speaker-diarization) – for diarization
  - [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) – for summarization and fine-tuning
  - [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) - for question answering
- **Libraries**:
  - `transformers`, `datasets`, `torch`, `sentencepiece`
  - `pyannote.audio`, `whisper`, `moviepy`, `pydub`, `ffmpeg-python`, `srt`
  - `evaluate`, `scikit-learn`, `tqdm`, `ffmpeg`, `yt-dlp`

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `smart_av_app.ipynb` | Main notebook for audio transcription, diarization, speaker tagging, and summarization |
| `Flan-T5-Finetuning.ipynb` | Fine-tuning FLAN-T5 on custom dialogue summarization task |
| `README.md` | Project documentation (this file) |
