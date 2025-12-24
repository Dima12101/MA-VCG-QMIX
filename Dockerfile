FROM python:3.9-slim

WORKDIR /app

# Установить зависимости системы
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Скопировать requirements
COPY requirements.txt .

# Установить Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать весь проект
COPY . .

# Создать директории для результатов
RUN mkdir -p experiments/results/plots experiments/logs

# Команда по умолчанию
CMD ["python", "main_run_all_scenarios.py"]
