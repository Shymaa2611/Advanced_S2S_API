from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from s2smodels import Base, Audio_segment, AudioGeneration
from pydub import AudioSegment
import os
from fastapi import FastAPI, Response
import torchaudio
from fastapi.responses import JSONResponse
from utils.prompt_making import make_prompt
from transformers import AutoProcessor, SeamlessM4Tv2Model
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from io import BytesIO
from pyannote.audio import Pipeline
import soundfile as sf
from fastapi_cors import CORS
from functools import lru_cache
DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

app = FastAPI()
""" 
origins = ["*"]

app.add_middleware(
    CORS,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
Base.metadata.create_all(engine)

@app.get("/")
def root():
    return {"message": "No result"}

#add audio segements in Audio_segment Table
def create_segment(start_time: float, end_time: float, audio: AudioSegment, type: str):
    session = Session()
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes = audio_bytes.getvalue()
    segment = Audio_segment(start_time=start_time, end_time=end_time, type=type, audio=audio_bytes)
    session.add(segment)
    session.commit()
    session.close()

    return {"status_code": 200, "message": "success"}

#add target audio to AudioGeneration Table
def generate_target(audio: AudioSegment):
    session = Session() 
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes = audio_bytes.getvalue()
    target_audio = AudioGeneration(audio=audio_bytes)
    session.add(target_audio)
    session.commit()
    session.close()

    return {"status_code": 200, "message": "success"}

@lru_cache(maxsize=None)
def load_segmentation_model():
    pipeline = Pipeline.from_pretrained(
     "pyannote/speaker-diarization-3.0",
    use_auth_token="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")
    return pipeline


"""
audio segmentation into speech and non-speech using segmentation model
"""
def audio_speech_nonspeech_detection(audio_url):
    pipeline=load_segmentation_model()
    diarization = pipeline(audio_url)
    speaker_regions=[]
    for turn, _,speaker in  diarization.itertracks(yield_label=True):
         speaker_regions.append({"start":turn.start,"end":turn.end})
    sound = AudioSegment.from_wav(audio_url)
    speaker_regions.sort(key=lambda x: x['start'])
    non_speech_regions = []
    for i in range(1, len(speaker_regions)):
        start = speaker_regions[i-1]['end'] 
        end = speaker_regions[i]['start']   
        if end > start:
            non_speech_regions.append({'start': start, 'end': end})
    first_speech_start = speaker_regions[0]['start']
    if first_speech_start > 0:
          non_speech_regions.insert(0, {'start': 0, 'end': first_speech_start})
    last_speech_end = speaker_regions[-1]['end']
    total_audio_duration = len(sound)  
    if last_speech_end < total_audio_duration:
            non_speech_regions.append({'start': last_speech_end, 'end': total_audio_duration})
    return speaker_regions,non_speech_regions

"""
save speech and non-speech segments in audio_segment table
"""
def split_audio_segments(audio_url):
    sound = AudioSegment.from_wav(audio_url)
    speech_segments, non_speech_segment = audio_speech_nonspeech_detection(audio_url)
    # Process speech segments
    for i, speech_segment in enumerate(speech_segments):
        start = int(speech_segment['start'] * 1000)  
        end = int(speech_segment['end'] * 1000)  
        segment = sound[start:end]
        create_segment(start_time=start/1000,
            end_time=end/1000,
            type="speech",audio=segment)
    # Process non-speech segments 
    for i, non_speech_segment in enumerate(non_speech_segment):
        start = int(non_speech_segment['start'] * 1000)  
        end = int(non_speech_segment['end'] * 1000)  
        segment = sound[start:end]
        create_segment(start_time=start/1000,
            end_time=end/1000,
            type="non-speech",audio=segment)


def detect_language(target_language:str):
    if target_language=="english":
        target_language="eng"
    elif target_language=="arabic":
        target_language="arz"
    elif target_language=="chinese":
        target_language="cmn"
    return target_language

@lru_cache(maxsize=None)
def load_seamless_model():
   processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
   model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
   return processor,model

""" 
 SeamlessM4T Model
source => speech in english
target => text in arabic

"""
def speech_text_translation(segment,target_language:str):
    target_language=detect_language(target_language)
    processor,model=load_seamless_model()
    audio, orig_freq =  torchaudio.load(segment)
    audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    output_tokens = model.generate(**audio_inputs, tgt_lang=target_language, generate_speech=False)
    translated_text= processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text

def make_prompt_audio(name,audio_path):
    make_prompt(name=name, audio_prompt_path=audio_path)



#text to speech using VALL-E-X model 
#@app.post("/text_to_speech/")  
def text_to_speech(segment_id, target_text, audio_prompt):
    preload_models()
    session = Session()
    segment = session.query(Audio_segment).get(segment_id)
    make_prompt_audio(name=f"audio_{segment_id}",audio_path=audio_prompt)
    audio_array = generate_audio(target_text,f"audio_{segment_id}")
    temp_file = BytesIO()
    sf.write(temp_file, audio_array, SAMPLE_RATE, format='wav')
    temp_file.seek(0)
    segment.audio = temp_file.getvalue()
    session.commit()
    session.close()
    temp_file.close()
    #os.remove(temp_file)

"""
reconstruct target audio using all updated segment
in audio_segment table and then remove all audio_Segment records
"""
def construct_audio():
    session = Session()
    # Should be ordered by start_time
    segments = session.query(Audio_segment).order_by('start_time').all()
    audio_files = []
    for segment in segments:
        audio_files.append(AudioSegment.from_file(BytesIO(segment.audio), format='wav'))
    target_audio = sum(audio_files, AudioSegment.empty())
    generate_target(audio=target_audio)
    
    # Delete all records in Audio_segment table
    session.query(Audio_segment).delete()
    session.commit()
    session.close()

"""
source  => english speech
target  => arabic speeech
"""

#@app.post("/speech_translation/")
def speech_to_speech_translation(audio_url,target_language:str):
    session=Session()
    target_text=None
    split_audio_segments(audio_url)
    #filtering by type
    speech_segments = session.query(Audio_segment).filter(Audio_segment.type == "speech").all()
    for segment in speech_segments:
        audio_data = segment.audio
        target_text=speech_text_translation(audio_data,target_language)
        if target_text is None:
            print("Target text is None.")
        else:
           segment_id = segment.id
           segment_duration = segment.end_time - segment.start_time
           if segment_duration <=15:
                text_to_speech(segment_id,target_text,segment.audio)
           else:
                audio_data=extract_15_seconds(segment.audio,segment.start_time,segment.end_time)
                text_to_speech(segment_id,target_text,audio_data)
                os.remove(audio_data)
    construct_audio()
    return JSONResponse(status_code=200, content={"status_code":"succcessfully"})
    

def get_audio(audio_url,target_language):
    speech_to_speech_translation(audio_url,target_language)
    session = Session()
    # Get target audio from AudioGeneration
    target_audio = session.query(AudioGeneration).order_by(AudioGeneration.id.desc()).first()
    if target_audio  is None:
        raise ValueError("No audio found in the database")
    audio_bytes = target_audio.audio
    file_path = "target_audio.wav"
    with open(file_path, "wb") as file:
        file.write(audio_bytes)
    session.close()
    return {"audio_url":file_path}

@app.get("/audio_segments/")
def get_all_audio_segments():
        session=Session()
        segments = session.query(Audio_segment).all()
        segment_dicts = []
        for segment in segments:
            if segment.audio is None:
                raise ValueError("No audio found in the database")

            audio_bytes = segment.audio
            file_path = f"segments//segment{segment.id}_audio.wav"
            with open(file_path, "wb") as file:
               file.write(audio_bytes)
            segment_dicts.append({
                "id": segment.id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "type": segment.type,
                "audio_url":file_path
            })
        session.close()
        return {"segments":segment_dicts}


def extract_15_seconds(audio_data, start_time, end_time):
    audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')
    start_ms = start_time * 1000  
    end_ms = min((start_time + 15) * 1000, end_time * 1000)  
    extracted_segment = audio_segment[start_ms:end_ms]
    temp_wav_path = "temp.wav"
    extracted_segment.export(temp_wav_path, format="wav")

    return temp_wav_path



if __name__=="main":
    
    audio_url="/content/audio_2.wav"
    get_audio(audio_url,"chinese")

    print("Done!")
	