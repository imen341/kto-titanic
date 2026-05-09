FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install uv
RUN uv sync --group training --group dev

CMD ["python", "src/titanic/training/main.py"]