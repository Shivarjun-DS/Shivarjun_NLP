FROM python:3.10.9-slim
RUN pip install -U pip
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
CMD ["python", "prediction.py"]
