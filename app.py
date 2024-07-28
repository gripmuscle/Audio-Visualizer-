import os
import tempfile
import math
import numpy as np
import cairo
import subprocess as sp
import streamlit as st
import tqdm
import shutil
from moviepy.editor import VideoFileClip
from io import BytesIO

# Function to read audio file and process it
def read_audio(audio, seek=None, duration=None):
    import soundfile as sf
    audio_file = BytesIO(audio.read())
    wav, sr = sf.read(audio_file, start=seek, duration=duration)
    return wav, sr

# Function to compute envelope
def envelope(wav, window, stride):
    frames = (len(wav) - window) // stride + 1
    env = np.zeros(frames)
    for i in range(frames):
        start = i * stride
        end = start + window
        env[i] = np.max(np.abs(wav[start:end]))
    return env

# Function to create the frames with transparency and rounded corners
def draw_env(envs, out, fg_colors, bg_color, size, radius):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)

    # Set the background to transparent
    ctx.set_source_rgba(*bg_color, 0)  # RGBA with 0 alpha for transparency
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    K = len(envs)  # Number of waves to draw (waves are stacked vertically)
    T = len(envs[0])  # Number of time steps
    pad_ratio = 0.1  # spacing ratio between 2 bars
    width = 1. / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(T):
        for i in range(K):
            half = 0.5 * envs[i][step]  # (semi-)height of the bar
            half /= K  # as we stack K waves vertically
            midrule = (1 + 2 * i) / (2 * K)  # midrule of i-th wave
            ctx.set_source_rgb(*fg_colors[i])
            ctx.move_to(pad + step * delta, midrule - half)
            ctx.line_to(pad + step * delta, midrule)
            ctx.stroke()
            ctx.set_source_rgba(*fg_colors[i], 0.8)
            ctx.move_to(pad + step * delta, midrule)
            ctx.line_to(pad + step * delta, midrule + 0.9 * half)
            ctx.stroke()

    # Draw rounded corners
    ctx.arc(0, 0, radius, 0, 2 * np.pi)
    ctx.fill()
    ctx.arc(size[0], 0, radius, 0, 2 * np.pi)
    ctx.fill()
    ctx.arc(0, size[1], radius, 0, 2 * np.pi)
    ctx.fill()
    ctx.arc(size[0], size[1], radius, 0, 2 * np.pi)
    ctx.fill()

    surface.write_to_png(out)

# Main visualization function
def visualize(audio,
              tmp,
              out,
              seek=None,
              duration=None,
              rate=60,
              bars=50,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              fg_color2=(.5, .3, .6),
              bg_color=(0, 0, 0),  # Transparent background
              size=(400, 400),
              stereo=False,
              radius=5,  # Default radius for rounded corners
              ):
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        st.error(f"Error reading audio: {err}")
        return
    wavs = []
    if stereo:
        assert wav.shape[0] == 2, 'stereo requires stereo audio file'
        wavs.append(wav[0])
        wavs.append(wav[1])
    else:
        wav = wav.mean(0)
        wavs.append(wav)

    for i, wav in enumerate(wavs):
        wavs[i] = wav / wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)
    envs = []
    for wav in wavs:
        env = envelope(wav, window, stride)
        env = np.pad(env, (bars // 2, 2 * bars))
        envs.append(env)

    duration = len(wavs[0]) / sr
    frames = int(rate * duration)
    smooth = np.hanning(bars)

    if not os.path.exists(tmp):
        os.makedirs(tmp)

    st.write("Generating the frames...")
    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)
        loc = pos - off
        denvs = []
        for env in envs:
            env1 = env[off * bars:(off + 1) * bars]
            env2 = env[(off + 1) * bars:(off + 2) * bars]

            maxvol = math.log10(1e-4 + env2.max()) * 10
            speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)
            w = sigmoid(speed * speedup * (loc - 0.5))
            denv = (1 - w) * env1 + w * env2
            denv *= smooth
            denvs.append(denv)
        draw_env(denvs, os.path.join(tmp, f"{idx:06d}.png"), (fg_color, fg_color2), bg_color, size, radius)

    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.name]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]

    st.write("Encoding the animation video...")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", os.path.join(tmp, "%06d.png")
    ] + audio_cmd + [
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-pix_fmt", "yuva420p",  # VP9 codec with alpha channel
        out
    ],
           check=True,
           cwd=tmp)

# Streamlit UI
st.title("Video Visualizer with Transparency")

uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_audio:
    tmp_dir = tempfile.mkdtemp()
    output_video_path = os.path.join(tmp_dir, "output_video.webm")

    visualize(uploaded_audio,
              tmp_dir,
              output_video_path,
              rate=30,
              bars=40,
              speed=4,
              time=0.5,
              oversample=2,
              fg_color=(0.8, 0.5, 0.5),
              fg_color2=(0.5, 0.8, 0.5),
              bg_color=(0, 0, 0),
              size=(800, 600),
              stereo=True,
              radius=10)

    with open(output_video_path, "rb") as video_file:
        st.download_button("Download Video", video_file, file_name="output_video.webm")

    # Clean up temporary files
    shutil.rmtree(tmp_dir)
