FROM continuumio/miniconda3

WORKDIR /app

COPY . .
RUN conda env create -n face-api-venv -f environment.yml

CMD ["sh", "-c", "conda run -n face-api-venv uvicorn face-api:app --host 0.0.0.0 --port 5000"]

