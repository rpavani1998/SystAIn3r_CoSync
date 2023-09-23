import io
import tempfile

import pyperclip
import streamlit as st
import whisper


class Transcribe:
    def __init__(self) -> None:
        with st.spinner("Loading model..."):
            self.model = whisper.load_model("base")

        self.options = whisper.DecodingOptions(fp16=False)

    

    def run(self, audio_file: io.BytesIO) -> None:
        # st.title("Scribe")

        # audio_file = st.file_uploader("Upload a file")

        # if audio_file is None:
        #     return

        # st.audio(audio_file)

        result = self.process_audio(audio_file)

        st.header("Result")
        st.write(result)  # type: ignore
        with open("output/transcript.txt", "w") as text_file:
            text_file.write(result)
        st.header("Copy to clipboard")
        if st.button("Copy"):
            pyperclip.copy(result) 

    @st.experimental_memo
    def process_audio(_self, audio_file: io.BytesIO) -> str:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(audio_file.read())

            result = _self.model.transcribe(tmp.name)

        return result["text"] 
