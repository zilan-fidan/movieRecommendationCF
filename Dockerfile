# Python 3.9 slim imajını kullan

FROM python:3.11-slim
# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Uygulamayı başlat
CMD ["python", "main.py"]
