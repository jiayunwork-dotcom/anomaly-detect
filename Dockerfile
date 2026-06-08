FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple .

COPY app/ app/

RUN mkdir -p /app/data

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.api.app:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
