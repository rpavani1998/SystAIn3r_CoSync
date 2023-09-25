# audio_recorder.py
import tempfile
import pyaudio
import wave
import streamlit as st

class AudioRecorder:
    def __init__(self):
        self.audio_frames = None
        self.audio_rate = None

    def record_audio(self, duration):
        audio_format = pyaudio.paInt16
        audio_channels = 1
        audio_rate = 44100
        audio_chunk = 1024

        p = pyaudio.PyAudio()

        if "recording_started" in st.session_state and st.session_state.recording_started:
            stream = p.open(format=audio_format,
                            channels=audio_channels,
                            rate=audio_rate,
                            input=True,
                            frames_per_buffer=audio_chunk)

            audio_frames = []
            for _ in range(0, int(audio_rate / audio_chunk * duration)):
                data = stream.read(audio_chunk)
                audio_frames.append(data)

            st.text("Recording complete!")

            stream.stop_stream()
            stream.close()
            p.terminate()

            self.audio_frames = audio_frames
            self.audio_rate = audio_rate

            return audio_frames, audio_rate

    def display_audio(self, audio_frames, audio_rate):
        st.audio(self.save_audio_to_tempfile(audio_frames, audio_rate), format="audio/wav")

    def is_audio_recorded(self):
        return self.audio_frames is not None

    def get_audio_data(self):
        return self.audio_frames, self.audio_rate

    def save_audio_to_tempfile(self, frames, sample_rate):
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_file.close()

        wf = wave.open(temp_audio_file.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return temp_audio_file.name
    
    def save_audio_to_uploaded_file(self, frames, sample_rate):
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_file.close()

        wf = wave.open(temp_audio_file.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # st.write("Audio saved as temporary WAV file:", temp_audio_file.name)

        # Upload the temporary WAV file and return it as an UploadedFile
        # uploaded_audio = st.sidebar.file_uploader("Uploaded Audio", type=["wav"], key="uploaded_audio")
        return temp_audio_file.name
