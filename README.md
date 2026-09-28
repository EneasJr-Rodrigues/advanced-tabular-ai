# 🧠 Advanced Tabular AI: Risco com Explicabilidade Esparsa e Calibração Conformal

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## Visão Geral
Este repositório contém a implementação de um sistema corporativo de Inteligência Artificial para dados tabulares voltado à Análise de Risco. Desenvolvido para superar o desempenho de modelos tradicionais (como XGBoost) através do uso de redes neurais profundas focadas em explicabilidade de "caixa-branca" (White-box AI).

A arquitetura resolve dois dos maiores problemas de IA no mercado financeiro e de risco:
1. **Falta de Explicabilidade:** Implementação de Atenção Esparsa (*Sparsemax*), que zera matematicamente variáveis irrelevantes, permitindo saber exatamente o porquê de cada decisão.
2. **Incerteza Oculta:** Integração de *Conformal Prediction*, garantindo um limite estatístico de cobertura para alertar quando o modelo não tem confiança suficiente para tomar uma decisão.

## 🏗️ Estrutura da Arquitetura
- **Camada de Treinamento (PyTorch):** Processamento com normalização Z-score e exportação unificada de artefatos (Model + Scalers + Encoders).
- **Camada de Inferência (FastAPI):** Microsserviço assíncrono para previsões e cálculo de matrizes de atenção.
- **Camada de Decisão (Streamlit):** Dashboard executivo com renderização de atenção matemática via `Plotly`.

---

## 🚀 Como Executar o Projeto Localmente

O ambiente foi isolado utilizando infraestrutura imutável via Docker, eliminando conflitos de dependências de hardware ou versão.

### Pré-requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando.
- Git instalado.

### Passo a Passo
1. Clone este repositório:
   ```bash
   git clone [https://github.com/SEU-USUARIO/advanced-tabular-ai.git](https://github.com/SEU-USUARIO/advanced-tabular-ai.git)
   cd advanced-tabular-ai
    ```

2. Suba o ecossistema (API + Dashboard) via Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Acesse os serviços no seu navegador:

* Dashboard Executivo (Streamlit): http://localhost:8501
* Documentação da API (Swagger): http://localhost:8000/docs

## 📚 Documentação Técnica Consultada

A fundamentação teórica e a implementação técnica deste motor basearam-se nas seguintes bibliografias e documentações oficiais:
* From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification (Martins & Astudillo): Base matemática para a camada de explicabilidade neural.

* A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification (Angelopoulos & Bates): Metodologia estatística para quantificação de incerteza em modelos de Machine Learning.

* [PyTorch Documentation](https://docs.pytorch.org/docs/2.14/index.html): Construção e otimização de tensores.

* [FastAPI Framework](https://fastapi.tiangolo.com/?utm_source=gemini): Arquitetura de microsserviços.

* [Streamlit Library](https://docs.streamlit.io/?utm_source=gemini): Desenvolvimento da interface reativa.