import pytest
from fastapi.testclient import TestClient
from main import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    engine = create_engine('sqlite:///:memory:')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()

def test_create_segment(test_db):
    start_time = 0.0
    end_time = 10.0
    type = "speech"
    audio = AudioSegment.silent(duration=1000)  
    response = create_segment(start_time, end_time, audio, type)
    assert response == {"status_code": 200, "message": "success"}
    created_segment = test_db.query(Audio_segment).filter_by(start_time=start_time, end_time=end_time, type=type).first()
    assert created_segment is not None
    #assert created_segment.start_time 
    #assert created_segment.end_time == end_time
    #assert created_segment.type == type



def test_generate_target(test_db):
    audio = AudioSegment.silent(duration=1000)  
    response = generate_target(audio)
    assert response == {"status_code": 200, "message": "success"}
    created_audio = test_db.query(AudioGeneration).order_by(AudioGeneration.id.desc()).first()
    assert created_audio is not None

def test_audio_speech_nonspeech_detection():
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    speech_segments, non_speech_segments = audio_speech_nonspeech_detection(audio_url)
    assert len(speech_segments) > 0
    assert len(non_speech_segments) > 0

def test_speech_text_translation():
    text="How are you today"
    target_language="arabic"
    target_text=speech_text_translation(text,target_language)
    assert target_text is not None



def test_speech_to_speech_translation():
    audio_url="C:\\Users\\dell\\Downloads\\Music\\audio.wav"
    target_language="arabic"
    response =speech_to_speech_translation(audio_url,target_language)
    assert response.status_code == 200
    assert response.json() == {"status_code":"succcessfully"}


def test_get_audio():
    client = TestClient(app)
    test_params = {
        "audio_url": "C:\\Users\\dell\\Downloads\\Music\\audio.wav",
        "target_language": "arabic"
    }
    response = client.get("/get_audio/", params=test_params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    assert response.headers["Content-Type"] == "audio/wav", "Response does not contain audio data"
    print("Test passed: Audio retrieved successfully")

