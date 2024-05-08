# SPEECH2SPEECH TRANSLATION 
 - Speech-to-speech translation (S2ST) aims at converting speech from one language into speech in    
   another.

## Details
 - S2ST is implemented using advanced approaches, such as  seamlessM4T to convert spoken words in  source language into text in target language and text-to-speech (TTS) synthesis to convert the translated text back into  speech. 

### AI Models
- segmentation model (V-segment)
- speech to text translation (seamlessM4T)
- text to speech (VALL-E-X)

![S2ST](api_process_image.jpg)

## Technologies
- FASTAPI
- sqlite database
- Natural Language processing
- Large Language Model
- Speech Processing
- Docker

## Framework 
- FASTAPI

## Usage
  git clone https://github.com/Shymaa2611/Advanced_S2S_API.git
  <br>
  cd S2S_API_FastAPI
  <br>
  pip install -r requirements.txt
  <br>
  uvicorn main:app --reload


### Running
 
  - http://127.0.0.1:8000/docs



### Docker
 
  - docker build -t s2simage .
  - docker run -d --name s2scontainer -p 80:80 s2simage


### Deploy

  




