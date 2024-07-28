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

# Function to create video from waveform
def create_waveform_video(envelope, output_path, frame_rate=24, width=1280, height=720, color='#FF0000', background_color='white', transparent_bg=False, rounded_bars=False, radius=0):
    try:
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
        # Plot the waveform and write each frame to the video
        plt.figure(figsize=(width/100, height/100), dpi=100)
        for i in range(len(envelope)):
            plt.clf()
            plt.fill_between(np.arange(len(envelope[:i])), envelope[:i], color=color)
            plt.xlim(0, len(envelope))
            plt.ylim(0, np.max(envelope))
            plt.gca().set_facecolor(background_color)
            if transparent_bg:
                plt.gca().patch.set_alpha(0)
            if rounded_bars:
                plt.gca().patch.set_radius(radius)
            plt.title('Audio Envelope')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            
            # Save plot as image
            plt.savefig('temp_frame.png', bbox_inches='tight')
            frame = cv2.imread('temp_frame.png')
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
        
        out.release()
        st.write("Video created successfully.")
    except Exception as e:
        st.error(f"Error creating video: {e}")

# Streamlit app
st.title("Audio Waveform Video Generator")

# Upload audio
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_audio:
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

            # Create video from waveform
            output_path = "waveform_video.webm"
            create_waveform_video(envelope, output_path, frame_rate=24, color=waveform_color, background_color=background_color, transparent_bg=transparent_bg, rounded_bars=rounded_bars, radius=radius)
            
            # Provide download link
            with open(output_path, "rb") as f:
                st.download_button("Download the generated video", f, file_name="waveform_video.webm")
else:
    st.warning("Please upload an audio file.")
