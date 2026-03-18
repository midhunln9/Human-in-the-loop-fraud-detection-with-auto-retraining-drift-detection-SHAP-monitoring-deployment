FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock /app

RUN uv sync 

COPY . /app

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]