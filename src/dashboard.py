import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Tabular AI | Conformal Prediction", layout="wide")

st.title("🧠 Inteligência Artificial Tabular Avançada")
st.markdown("### Análise de Risco com Explicabilidade Esparsa e Calibração Conformal")
st.markdown("---")

# Layout principal: Coluna da esquerda (Inputs) e Coluna da direita (Resultados)
col_inputs, col_resultados = st.columns([1, 2])

with col_inputs:
    st.header("Dados do Indivíduo")
    
    # Inputs Contínuos
    age = st.slider("Idade", 18, 90, 35)
    hours_per_week = st.slider("Horas de Trabalho/Semana", 1, 100, 40)
    capital_gain = st.number_input("Ganho de Capital ($)", 0, 100000, 0)
    capital_loss = st.number_input("Perda de Capital ($)", 0, 5000, 0)
    education_num = st.slider("Anos de Estudo", 1, 16, 10)
    fnlwgt = st.number_input("Peso Amostral (fnlwgt)", 10000, 1500000, 150000)
    
    # Inputs Categóricos
    workclass = st.selectbox("Classe de Trabalho", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay"])
    education = st.selectbox("Nível de Educação", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate"])
    marital_status = st.selectbox("Estado Civil", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed"])
    occupation = st.selectbox("Ocupação", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving"])
    relationship = st.selectbox("Relacionamento", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Etnia", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.selectbox("Sexo", ["Female", "Male"])
    native_country = st.selectbox("País de Origem", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Other"])

    analisar_btn = st.button("Executar Inferência da IA", use_container_width=True, type="primary")

with col_resultados:
    if analisar_btn:
        # Montando o JSON idêntico ao que vimos no Swagger
        payload = {
            "age": float(age),
            "fnlwgt": float(fnlwgt),
            "education_num": float(education_num),
            "capital_gain": float(capital_gain),
            "capital_loss": float(capital_loss),
            "hours_per_week": float(hours_per_week),
            "workclass": workclass,
            "education": education,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "native_country": native_country
        }

        with st.spinner('Processando matrizes de atenção e limites conformais...'):
            try:
                # Enviando para a nossa API FastAPI local
                # response = requests.post("http://localhost:8000/predict", json=payload)
                api_url = os.getenv("API_URL", "http://localhost:8000/predict")
                response = requests.post(api_url, json=payload)

                response.raise_for_status()
                resultado = response.json()
                
                pred = resultado["prediction"]
                explicabilidade = resultado["explicabilidade"]
                
                # Recuperando as variáveis da resposta da API
                probabilidade = pred["probabilidade_fraude_ou_alto_risco"]
                decisao = pred["decisao_sugerida"]
                alerta = pred["alerta_conformal_prediction"]
                
                # 1. Painel de Decisão: Velocímetro de Risco (Gauge Chart)
                st.header("Resultado da Análise")
                
                # Exibição visual de Alerta Conformal (Incerteza)
                if alerta:
                    st.error(f"🚨 **ALERTA CONFORMAL ATIVADO:** A IA detectou uma incerteza matemática acima do limite seguro. Decisão sugerida: **{decisao}**.")
                else:
                    if probabilidade > 0.5:
                        st.warning(f"⚠️ **ALTO RISCO DETECTADO.** Decisão sugerida: **{decisao}**.")
                    else:
                        st.success(f"✅ **BAIXO RISCO.** Decisão sugerida: **{decisao}**.")
                
                # Criando o Velocímetro (Gauge) com Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilidade * 100,
                    number={'suffix': "%", 'valueformat': ".1f"},
                    title={'text': "Probabilidade Calculada pela Rede Neural"},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                # Ajusta a altura e as margens do gráfico
                fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("---")
                
                # 2. Gráfico de Explicabilidade: Visão Dupla (Barras + Heatmap)
                st.header("Raio-X do Modelo: Explicabilidade Esparsa")
                st.markdown("A inteligência do modelo pode ser vista em dois formatos. Os valores **zerados** indicam que a matemática eliminou a variável da equação.")
                
                features_names = ["Age", "Fnlwgt", "Edu-Num", "Cap-Gain", "Cap-Loss", "Hours", "Workclass", "Education", "Marital", "Occupation", "Relationship", "Race", "Sex", "Country"]
                pesos = explicabilidade["pesos_atencao_esparsa"]
                
                # Exibição 1: Gráfico de Barras (Clareza de Grandeza)
                df_plot = pd.DataFrame({
                    "Variáveis": features_names,
                    "Importância Matemática": pesos
                })
                # Usando o bar_chart nativo do Streamlit que é muito elegante
                st.bar_chart(df_plot.set_index("Variáveis"), height=250)
                
                st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
                
                # Exibição 2: Mapa de Calor (Visão de Temperatura)
                # fig_heat = px.imshow(
                #     [pesos],
                #     labels=dict(x="Variáveis", y="Importância", color="Atenção Esparsa"),
                #     x=features_names,
                #     y=["Temperatura"],
                #     color_continuous_scale="Blues",
                #     aspect="auto"
                # )
                
                # fig_heat.update_xaxes(side="bottom")
                # fig_heat.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
                
                # # Adiciona os números em cima dos quadradinhos
                # for i, peso in enumerate(pesos):
                #     fig_heat.add_annotation(
                #         x=i, y=0, text=f"{peso:.2f}", showarrow=False,
                #         font=dict(color="white" if peso > 0.08 else "black", size=10)
                #     )
                
                # st.plotly_chart(fig_heat, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao conectar com a API: {e}")
    else:
        st.info("Preencha os dados ao lado e clique em 'Executar Inferência' para ver o motor em ação.")