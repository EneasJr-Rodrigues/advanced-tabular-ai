# Usa uma imagem oficial leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala ferramentas do sistema necessárias para compilar pacotes Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copia os requisitos e instala as dependências sem gerar cache (economiza espaço)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta de códigos e a pasta de artefatos treinados
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Muda o diretório raiz para dentro da pasta src para execução
WORKDIR /app/src