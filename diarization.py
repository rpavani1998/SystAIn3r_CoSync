# import whisper
# import datetime
# import torch
# import pyannote.audio
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# from pyannote.audio import Audio
# from pyannote.core import Segment
# import wave
# import contextlib
# from sklearn.cluster import AgglomerativeClustering
# import numpy as np

# class SpeakerDiarization:
#     def __init__(self, model=None, model_size='base', num_speakers=2):
#         self.model_size = model_size
#         if model == None:
#             self.model = self.load_model()
#         self.model = model
#         self.NUM_SPEAKERS = num_speakers

#     def load_model(self):
#         return whisper.load_model(self.model_size)

#     def transcribe_audio(self, model, audio_path):
#         print(audio_path)
#         result = model.transcribe(audio_path)
#         return result["segments"]

#     def load_audio(self, audio_path):
#         audio = Audio()
#         with contextlib.closing(wave.open(audio_path, 'r')) as f:
#             frames = f.getnframes()
#             rate = f.getframerate()
#             duration = frames / float(rate)
#         return audio, duration

#     def segment_embedding(self, embedding_model, audio_path, segment):
#         start = segment["start"]
#         end = min(self.duration, segment["end"])
#         clip = Segment(start, end)
#         waveform, sample_rate = self.audio.crop(audio_path, clip)
#         return embedding_model(waveform[None])

#     def compute_embeddings(self, segments, audio_path, embedding_model):
#         embeddings = np.zeros(shape=(len(segments), 192))
#         for i, segment in enumerate(segments):
#             embeddings[i] = self.segment_embedding(embedding_model, audio_path, segment)
#         return np.nan_to_num(embeddings)

#     def cluster_speakers(self, embeddings):
#         clustering = AgglomerativeClustering(self.NUM_SPEAKERS).fit(embeddings)
#         labels = clustering.labels_
#         return labels

#     def generate_transcript(self, segments, labels):
#         transcript = []
#         for i, segment in enumerate(segments):
#             if i == 0 or labels[i - 1] != labels[i]:
#                 transcript.append("\nSPEAKER {} {}".format(labels[i] + 1, str(self.time(segment["start"]))))
#             transcript.append(segment["text"][1:])
#         return " ".join(transcript)

#     def process_audio(self, audio_path):
#         # Load the whisper model
#         model = self.model

#         # Transcribe audio using the model
#         segments = self.transcribe_audio(model, audio_path)

#         # Load audio using pyannote
#         self.audio, self.duration = self.load_audio(audio_path)

#         # Create the embedding model
#         embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")

#         # Compute speaker embeddings
#         embeddings = self.compute_embeddings(segments, audio_path, embedding_model)

#         # Cluster speakers
#         labels = self.cluster_speakers(embeddings)

#         # Generate and save the transcript
#         transcript = self.generate_transcript(segments, labels)
#         with open("transcript_speakers.txt", "w") as f:
#             f.write(transcript)

# if __name__ == "__main__":
#     # Define the path to the audio file
#     audio_path = '/Users/pavanirajula/Documents/Hackathons/SystAIn3r/Speaker-Diarization/samples/recording_2.wav'
    
#     # Create an instance of the SpeakerDiarization class
#     diarization = SpeakerDiarization(model_size='base')

#     # Process the audio file
#     diarization.process_audio(audio_path)


import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb")

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

num_speakers = 2 

language = 'English'
model_size = 'base'
model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'

# if path[-3:] != 'wav':
#   subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
path = 'recording_2.wav'
 
model = whisper.load_model(model_size)

result = model.transcribe(path)
segments = result["segments"]

with contextlib.closing(wave.open(path,'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

audio = Audio()

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
    return datetime.timedelta(seconds=round(secs))

f = open("transcript_speakers.txt", "w")

for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][1:] + ' ')
f.close()