# 🧑‍⚖️ AI-Powered Courtroom Monitoring and Summarization System

An end-to-end AI pipeline that transcribes and summarizes courtroom audio using Automatic Speech Recognition (ASR), speaker diarization, and transformer-based summarization.

> 🎓 Accepted at IEEE IC3 2025 (~35% acceptance rate) from over 480 global submissions.

---

## 🚀 Project Overview

This project aims to automate courtroom documentation using advanced AI techniques by converting raw multi-speaker courtroom audio into structured, human-readable summaries.

### 🔍 Key Features:
- 🎙️ **Automatic Speech Recognition (ASR)** using Google Web Speech API
- 🧑‍⚖️ **Speaker Diarization** using AssemblyAI to identify "who said what"
- 📄 **Abstractive Summarization** using Facebook's `bart-large-cnn`
- 📝 Outputs concise, speaker-attributed legal summaries
- ⏱️ Near real-time processing with ~15s overhead per audio file

---

## 🧱 Tech Stack

| Category            | Tools & Libraries                                    |
|---------------------|------------------------------------------------------|
| Language            | Python 3                                             |
| ASR                 | Google Web Speech API, `speech_recognition`          |
| Diarization         | [AssemblyAI](https://www.assemblyai.com/) API        |
| Summarization       | `facebook/bart-large-cnn` via Hugging Face Transformers |
| Audio Processing    | `Pydub`, `SoundDevice`, `NumPy`, `SciPy`             |
| Models Runtime      | PyTorch                                              |

---

![image](https://github.com/user-attachments/assets/7acc0705-e3f1-48f8-acbd-e4a4b552b069)


## 🗂️ Dataset Details

### 🎧 Audio Dataset
- A custom dataset consisting of **courtroom-style audio clips**.
- Each clip is **30 seconds to 5 minutes long**, containing multiple speakers (Judge, Prosecutor, Witness).
- Audio is in `.wav` format, sampled at **16 kHz**, mono-channel.
- Some audio is synthetically generated using **text-to-speech (TTS)** to simulate controlled legal dialogue for consistent evaluation.

### 📝 Ground Truth Transcripts
- Manual transcripts created for each clip to compute Word Error Rate (WER) and ROUGE metrics.
- Each transcript includes speaker roles, helping evaluate diarization accuracy.

### 📊 Model Training Data (Pretrained)
| Component        | Training Dataset         | Used For                         |
|------------------|--------------------------|----------------------------------|
| ASR (Google)     | Proprietary (cloud model)| Speech-to-text conversion        |
| Diarization      | Proprietary (AssemblyAI) | Speaker embedding + clustering   |
| Summarizer (BART)| CNN/DailyMail dataset    | Pretrained summarization model   |

---

## 📂 Repository Structure

-speech_text.py – ASR + Speaker Diarization using AssemblyAI

-legal_llm_summarizer.py – Abstractive summarization using BART

-diarized_summarizer.py – Batch summarization of transcript chunks

-/Speech_to_Text/ – Folder containing diarized transcript files

-summaries_output.csv – Final summarized outputs (as CSV)

-README.md – Project documentation and overview

📊 Evaluation Metrics & Results

| Component         | Metric                | Value                      | Remarks                                     |
| ----------------- | --------------------- | -------------------------- | ------------------------------------------- |
| **ASR**           | Word Error Rate (WER) | **16.41%**                 | Accurate transcription in noisy legal audio |
| **Diarization**   | Purity                | **0.91**                   | High speaker labeling accuracy              |
|                   | Coverage              | **0.89**                   | Most of speech assigned to a speaker        |
| **Summarization** | ROUGE-1 Precision     | **97.13%**                 | Key terms preserved in summary              |
|                   | ROUGE-1 Recall        | **26.85%**                 | Partial coverage of full transcript         |
|                   | ROUGE-1 F1 Score      | **0.503**                  | Balanced summary quality                    |
|                   | Compression Ratio     | **0.2765**                 | Summary is \~72% shorter than original      |
| **Pipeline**      | Latency               | **\~audio duration + 15s** | Near real-time                              |

![image](https://github.com/user-attachments/assets/123e56b9-2ea5-4c8b-94b9-04b396b78297)

