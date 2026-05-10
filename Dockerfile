FROM python:3.11-slim

WORKDIR /app

COPY index_embedded.html ./index.html

EXPOSE 7860

CMD ["python", "-m", "http.server", "7860"]
