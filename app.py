import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pydub import AudioSegment
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
def create_waveform_video(envelope, output_path, frame_rate=24, width=500, height=200, color='#FF0000', background_color='white', transparent_bg=False, rounded_bars=False, bar_thickness=5):
    try:
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
        # Create a frame for each point in the envelope
        for i in range(len(envelope)):
            # Create a new figure for each frame
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.fill_between(np.arange(len(envelope[:i])), envelope[:i], color=color, linewidth=bar_thickness)
            ax.set_xlim(0, len(envelope))
            ax.set_ylim(0, np.max(envelope))
            ax.set_facecolor(background_color)
            ax.set_title('Audio Envelope')
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')

            if transparent_bg:
                fig.patch.set_alpha(0)  # Set figure background to transparent
            
            # Save the plot as an image
            plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # Read the saved image and add it to the video
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
                waveform_color = st.color_picker("Waveform Color", "#FF0000")
                background_color = st.color_picker("Background Color", "#FFFFFF")
                transparent_bg = st.checkbox("Transparent Background", value=False)
                rounded_bars = st.checkbox("Rounded Bars", value=False)
                
                if rounded_bars:
                    bar_thickness = st.slider("Bar Thickness", 1, 20, 5)
                else:
                    bar_thickness = 5
                
                frame_rate = st.slider("Frame Rate", 1, 60, 24)
                resolution = st.selectbox(
                    "Select Video Resolution",
                    ["500x200", "640x480", "1280x720", "1920x1080"],
                    index=0
                )
                resolution_width, resolution_height = map(int, resolution.split('x'))
            
            # Create video from waveform
            output_path = "waveform_video.webm"
            create_waveform_video(
                envelope, 
                output_path, 
                frame_rate=frame_rate, 
                width=resolution_width, 
                height=resolution_height, 
                color=waveform_color, 
                background_color=background_color, 
                transparent_bg=transparent_bg, 
                rounded_bars=rounded_bars, 
                bar_thickness=bar_thickness
            )
            
            # Provide download link
            with open(output_path, "rb") as f:
                st.download_button("Download the generated video", f, file_name="waveform_video.webm")
else:
    st.warning("Please upload an audio file.")
