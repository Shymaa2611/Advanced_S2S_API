FROM python:3.11
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install  -r /code/requirements.txt
COPY ./s2smodels.py /code/
COPY ./whisper /code/
COPY ./macros.py /code/
COPY ./sql_app.db /code/
COPY ./utils /code/utils/
COPY ./modules /code/modules/
COPY ./models /code/models/
COPY ./data /code/data/
COPY ./prompts /code/prompts/
COPY ./customs /code/customs/
COPY ./main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

