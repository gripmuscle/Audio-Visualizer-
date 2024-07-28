import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pydub import AudioSegment
from pydub.silence import split_on_silence
from PIL import Image

# Function to read audio and ensure it's stereo
def read_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        
        # Check if the audio is mono
        if audio.channels == 1:
            st.warning("Audio is mono. Converting to stereo.")
            audio = audio.set_channels(2)
        
        return audio
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
        return None

# Function to calculate audio envelope
def calculate_audio_envelope(audio):
    try:
        samples = np.array(audio.get_array_of_samples())
        envelope = np.abs(samples)  # Simplistic envelope calculation
        return envelope
    except Exception as e:
        st.error(f"Error calculating audio envelope: {e}")
        return None

# Function to generate frames from video
def generate_frames(video_path, frame_rate=24):
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with st.spinner("Extracting frames..."):
            for _ in range(total_frames):
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Error generating frames from video: {e}")
        return []

# Function to create video from frames
def create_video(frames, output_path, frame_rate=24):
    try:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        with st.spinner("Creating video..."):
            for frame in frames:
                out.write(frame)
        out.release()
    except Exception as e:
        st.error(f"Error creating video: {e}")

# Streamlit app
st.title("Audio and Video Processing")

# Upload video and audio
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_video and uploaded_audio:
    # Read and process audio
    audio = read_audio(uploaded_audio)
    if audio:
        envelope = calculate_audio_envelope(audio)
        if envelope is not None:
            st.write("Audio envelope calculated successfully.")
            
            # Customize waveform
            with st.sidebar:
                st.header("Waveform Customization")
                waveform_color = st.color_picker("Waveform Bars Color", "#FF0000")
                background_color = st.color_picker("Background Color", "#FFFFFF")
                transparent_bg = st.checkbox("Transparent Background", value=False)
                rounded_bars = st.checkbox("Rounded Bars", value=False)
                
                if rounded_bars:
                    radius = st.slider("Radius of Bars", 0, 20, 5)
                else:
                    radius = 0

            # Plot audio envelope
            plt.figure(figsize=(10, 4))
            plt.fill_between(np.arange(len(envelope)), envelope, color=waveform_color)
            plt.gca().set_facecolor(background_color)
            if transparent_bg:
                plt.gca().patch.set_alpha(0)
            if rounded_bars:
                plt.gca().patch.set_radius(radius)
            plt.title('Audio Envelope')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            st.pyplot(plt)

    # Video processing
    with st.sidebar:
        st.header("Video Processing")
        video_resolution = st.selectbox(
            "Select Video Resolution",
            ["500x200", "640x480", "1280x720", "1920x1080"],
            index=0
        )
        resolution_width, resolution_height = map(int, video_resolution.split('x'))
        frame_radius = st.slider("Frame Radius", 0, 20, 5)

    frames = generate_frames(uploaded_video)
    if frames:
        st.write("Frames extracted successfully.")
        
        # Create output video
        output_path = "output_video.webm"
        create_video(frames, output_path, frame_rate=24)
        
        # Provide download link
        with open(output_path, "rb") as f:
            st.download_button("Download the generated video", f, file_name="output_video.webm")
else:
    st.warning("Please upload both a video and an audio file.")
