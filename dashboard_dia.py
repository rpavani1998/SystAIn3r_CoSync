import streamlit as st

from transcribe import Transcribe
from audio_recorder import AudioRecorder
import io
# from diarization import SpeakerDiarization
from text_analytics import TextAnalyzer
from plots import Plotter
import whisper
import datetime
import base64
from audio_recorder_streamlit import audio_recorder
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb")

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def time(secs):
    return datetime.timedelta(seconds=round(secs))

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('images/bg.png')

transcribe = Transcribe()
st.title("Dashboard")

uploaded_file = None

st.sidebar.title(" Record/Upload Audio")
option = st.sidebar.radio("Select an option:", ("Record Audio", "Upload Audio File"))
recorder = AudioRecorder()

if option == "Record Audio":
        duration = st.sidebar.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
        recording_started = st.sidebar.button("Start Recording")

        if recording_started:
            st.write("Recording...")
            st.session_state.recording_started = True

        if "recording_started" in st.session_state and st.session_state.recording_started:
            audio_frames, audio_rate = recorder.record_audio(duration)
            # recorder.display_audio(audio_frames, audio_rate)
            recorded_file = recorder.save_audio_to_uploaded_file( audio_frames, audio_rate)
            with open(recorded_file, 'rb') as f:
                file_contents = f.read()

            # Create an in-memory binary stream
            uploaded_file = io.BytesIO(file_contents)
            # Display the uploaded file
            st.audio(uploaded_file, format="audio/wav")
            uploaded_file.name = recorded_file
            # st.write("uploaded_file", uploaded_file, type(uploaded_file))
            # uploaded_file = io.BytesIO(b''.join(audio_frames))
            # uploaded_file = "temp.wav" 

            # audio_bytes = audio_recorder()
            # # if audio_bytes:
            # #     st.audio(audio_bytes, format="audio/wav")
            # audio_bytes = audio_recorder(pause_threshold=5.0, sample_rate=41_000)

elif option == "Upload Audio File":
        uploaded_file = st.sidebar.file_uploader("Upload an audio file (WAV format recommended)", type=["wav"])
        # st.write("uploaded_file", uploaded_file, type(uploaded_file))
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")

if uploaded_file is not None:
    tab1, tab2 = st.tabs(["Overview", "Speakerwise"])
    with tab1:
        transcribe.run(uploaded_file)
        analyzer = TextAnalyzer()
        with open("output/transcript.txt", "r") as f:
            text_to_analyze = f.read()
        scores = analyzer.analyze_text(text_to_analyze)
        # sentiment_score = 0.416193
        # emotion_dict = {
        #     "sadness": 0.513162,
        #     "joy": 0.225002,
        #     "fear": 0.063327,
        #     "disgust": 0.039539,
        #     "anger": 0.052626
        # }
        # st.write("Sentiment Label:", scores['sentiment']['label'])
        # st.write("Sentiment Score:", scores['sentiment']['score'])
        # st.write("Emotion:", scores['emotion'])
        plotter = Plotter(scores['sentiment']['score'], scores['emotion'])
        # plotter = Plotter(sentiment_score,emotion_dict)
        cc1,cc2=st.columns([1,1])
        with cc1:
            st.plotly_chart(plotter.create_pie_chart(), use_container_width=True) 
        with cc2:
            st.plotly_chart( plotter.create_radar_chart(), use_container_width=True)
        st.plotly_chart(plotter.create_stacked_bar_chart(), use_container_width=True)
    
        with tab2:
            num_speakers = int(st.text_input("Enter the no. of participants", "2", key="placeholder",))
            model = transcribe.model
            result = model.transcribe(uploaded_file.name)
            segments = result["segments"]
            path = str(uploaded_file.name)
            # st.write(path)
            with contextlib.closing(wave.open(path,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)

            audio = Audio()
            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)

            embeddings = np.nan_to_num(embeddings)
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
            f = open("transcript_speakers.txt", "w")
            st.header("Speaker Diarization")
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    st.write('**' + segment["speaker"] + ' ' + str(time(segment["start"])) + '**')
                    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
                f.write(segment["text"][1:] + ' ')
                st.write(segment["text"][1:] + ' ')

            f.close()

        
       
