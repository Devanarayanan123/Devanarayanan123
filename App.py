import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from scipy.io.wavfile import write as wav_write
from scipy.fft import rfft, rfftfreq
from scipy.signal import resample
import soundfile as sf
import sympy as sp
import yt_dlp
from pydub import AudioSegment


st.set_page_config(page_title="Music ↔ Equation Studio", page_icon="🎼", layout="wide")


@dataclass
class EquationResult:
    sample_rate: int
    duration: float
    dominant_terms: List[Tuple[float, float, float]]  # amplitude, frequency, phase
    envelope_poly: np.ndarray


def load_audio_bytes(file_bytes: bytes, suffix: str = ".wav") -> Tuple[np.ndarray, int]:
    """Load many audio formats into mono float waveform."""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(file_bytes)
            temp_path = Path(temp.name)
        audio = AudioSegment.from_file(temp_path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
        peak = np.max(np.abs(samples)) if len(samples) else 1.0
        return samples / max(peak, 1.0), audio.frame_rate
    except Exception:
        pass

    data, sample_rate = sf.read(io.BytesIO(file_bytes), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data)) if len(data) else 1.0
    return data / max(peak, 1.0), sample_rate


def download_audio_from_url(url: str) -> Tuple[bytes, str]:
    """Download best available audio from URL/YouTube and return bytes + extension."""
    with tempfile.TemporaryDirectory() as td:
        outtmpl = str(Path(td) / "source.%(ext)s")
        opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
            file_bytes = path.read_bytes()
            return file_bytes, path.suffix or ".mp3"


def audio_to_equation(signal: np.ndarray, sr: int, n_terms: int = 8) -> EquationResult:
    if len(signal) < 16:
        raise ValueError("Audio is too short to analyze.")

    target_sr = 22050
    if sr != target_sr:
        signal = resample(signal, int(len(signal) * target_sr / sr))
        sr = target_sr

    n = len(signal)
    duration = n / sr
    yf = rfft(signal)
    xf = rfftfreq(n, 1 / sr)
    power = np.abs(yf)

    power[0] = 0
    idx = np.argpartition(power, -n_terms)[-n_terms:]
    idx = idx[np.argsort(power[idx])[::-1]]

    dominant_terms = []
    for i in idx:
        amplitude = 2 * np.abs(yf[i]) / n
        frequency = xf[i]
        phase = np.angle(yf[i])
        dominant_terms.append((float(amplitude), float(frequency), float(phase)))

    env = np.abs(signal)
    t = np.linspace(0, duration, n)
    coeffs = np.polyfit(t, env, deg=4)

    return EquationResult(sr, duration, dominant_terms, coeffs)


def format_equation(eq: EquationResult) -> str:
    t = sp.symbols("t", real=True)
    expr = 0
    for amp, freq, phase in eq.dominant_terms:
        expr += amp * sp.sin(2 * sp.pi * freq * t + phase)
    expr = sp.simplify(expr)

    env_expr = sum(float(c) * (t ** i) for i, c in enumerate(eq.envelope_poly[::-1]))
    env_expr = sp.simplify(sp.Abs(env_expr))

    return (
        "x(t) = "
        + sp.sstr(expr)
        + "\n\n"
        + "Envelope e(t) = "
        + sp.sstr(env_expr)
        + "\n\n"
        + "Approximation: y(t) = x(t) * e(t)"
    )


def function_to_audio(expr_text: str, duration: float, sr: int = 44100) -> Tuple[np.ndarray, int]:
    t = sp.symbols("t", real=True)
    expr = sp.sympify(expr_text)
    fn = sp.lambdify(t, expr, modules=["numpy"])

    ts = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.nan_to_num(fn(ts), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    peak = np.max(np.abs(y)) if len(y) else 1.0
    if peak > 0:
        y = y / peak
    fade_len = int(min(sr * 0.02, len(y) // 2))
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        y[:fade_len] *= fade
        y[-fade_len:] *= fade[::-1]
    return y, sr


def to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    scaled = np.int16(np.clip(samples, -1, 1) * 32767)
    buf = io.BytesIO()
    wav_write(buf, sr, scaled)
    return buf.getvalue()


st.title("🎼 Music ↔ Mathematical Equation Studio")
st.caption("Convert music to approximate mathematical functions, and generate music from custom functions.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Music ➜ Equation")
    source_type = st.radio("Input source", ["Upload File", "URL / YouTube Link"], horizontal=True)

    audio_bytes: Optional[bytes] = None
    suffix = ".wav"

    if source_type == "Upload File":
        file = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "aiff", "opus"],
        )
        if file:
            audio_bytes = file.read()
            suffix = Path(file.name).suffix.lower() or ".wav"
    else:
        url = st.text_input("Paste URL (YouTube, SoundCloud, direct file link, etc.)")
        if st.button("Fetch Audio From URL") and url.strip():
            with st.spinner("Downloading and extracting audio..."):
                try:
                    audio_bytes, suffix = download_audio_from_url(url.strip())
                    st.success("Audio fetched successfully.")
                except Exception as exc:
                    st.error(f"Could not fetch audio: {exc}")

    if audio_bytes and st.button("Convert Music to Equation"):
        try:
            signal, sr = load_audio_bytes(audio_bytes, suffix)
            result = audio_to_equation(signal, sr)
            equation_text = format_equation(result)

            st.audio(audio_bytes)
            st.code(equation_text, language="text")
            st.write(f"Sample rate used: **{result.sample_rate} Hz**, duration: **{result.duration:.2f}s**")

            freq_table = [
                {"Amplitude": round(a, 5), "Frequency (Hz)": round(f, 2), "Phase (rad)": round(p, 4)}
                for a, f, p in result.dominant_terms
            ]
            st.dataframe(freq_table, use_container_width=True)
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")

with col2:
    st.subheader("Equation ➜ Music")
    st.markdown(
        "Examples: `sin(2*pi*440*t)`, `0.5*sin(2*pi*220*t)+0.4*sin(2*pi*330*t)`, `sin(2*pi*(220+40*t)*t)`"
    )
    expr_text = st.text_area("Enter function y(t)", value="0.6*sin(2*pi*440*t) + 0.3*sin(2*pi*660*t)", height=130)
    duration = st.slider("Duration (seconds)", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
    sr = st.selectbox("Sample rate", [22050, 32000, 44100, 48000], index=2)

    if st.button("Generate Music from Function"):
        try:
            samples, out_sr = function_to_audio(expr_text, duration, sr)
            wav_bytes = to_wav_bytes(samples, out_sr)
            st.audio(wav_bytes, format="audio/wav")
            st.download_button("Download Generated WAV", data=wav_bytes, file_name="function_music.wav", mime="audio/wav")
            st.success("Music generated.")
        except Exception as exc:
            st.error(f"Invalid function or generation error: {exc}")

st.markdown("---")
st.markdown(
    "**Format support:** WAV, MP3, FLAC, OGG, AAC, AIFF, OPUS, M4A and more (best results with FFmpeg available)."
)
