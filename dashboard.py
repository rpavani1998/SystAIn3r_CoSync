import streamlit as st

from transcribe import Transcribe
from audio_recorder import AudioRecorder
import io
# from diarization import SpeakerDiarization
from text_analytics import TextAnalyzer
from plots import Plotter

import base64

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
            st.session_state.recording_started = True

        if "recording_started" in st.session_state and st.session_state.recording_started:
            audio_frames, audio_rate = recorder.record_audio(duration)
            recorder.display_audio(audio_frames, audio_rate)
            uploaded_file = recorder.save_audio_to_tempfile( audio_frames, audio_rate)
            # uploaded_file = io.BytesIO(b''.join(audio_frames))
            # uploaded_file = "temp.wav" 

elif option == "Upload Audio File":
        uploaded_file = st.sidebar.file_uploader("Upload an audio file (WAV format recommended)", type=["wav"])
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
    
        # with tab2:
        #     diarization = SpeakerDiarization(scribe.model)
        #     diarization.process_audio(uploaded_file.name)
        
       
