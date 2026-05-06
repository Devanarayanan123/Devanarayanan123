# Music ↔ Equation Studio

An advanced, user-friendly Streamlit application that converts:

1. **Music to mathematical equations/functions** (Fourier-style approximation + envelope)
2. **Mathematical functions back to music**

It supports **uploaded audio files** and **URL/YouTube links** as inputs for music-to-equation conversion, and handles more than WAV (e.g., MP3, FLAC, OGG, AAC, M4A, AIFF, OPUS).

## Features

- 🎵 Upload audio in multiple formats and convert to an approximate equation.
- 🔗 Paste URL/YouTube links to fetch audio and convert to equation.
- 📈 Outputs dominant sine components (amplitude/frequency/phase table).
- ∑ Generates audio from custom mathematical function `y(t)`.
- 💾 Listen and download generated waveform as WAV.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL (typically `http://localhost:8501`).

## Notes

- For the widest format support, install **FFmpeg** on your machine.
- Equation extraction is an approximation intended for creative and educational use.
- 
