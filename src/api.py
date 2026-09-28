import os
import glob
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Importando a nossa arquitetura
from model.model import TabularModel

app = FastAPI(title="Advanced Tabular AI API", version="1.0")

# Variáveis globais para armazenar o estado do modelo na memória RAM do servidor
model = None
artifact = None
device = torch.device("cpu") # Para inferência, CPU costuma ser mais barata e suficiente

# 1. Definindo o Contrato de Dados (O que a API espera receber no JSON)
class InferenciaRequest(BaseModel):
    # Contínuas
    age: float
    fnlwgt: float
    education_num: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    # Categóricas
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

@app.on_event("startup")
def carregar_modelo():
    global model, artifact
    print("Iniciando servidor e buscando artefatos...")
    
    # Busca o arquivo .pt mais recente na pasta artifacts
    arquivos = glob.glob("../artifacts/*.pt")
    if not arquivos:
        # Tenta buscar a partir do diretório atual (dependendo de onde rodarmos o uvicorn)
        arquivos = glob.glob("artifacts/*.pt")
        if not arquivos:
            raise RuntimeError("Nenhum artefato de modelo encontrado. Treine o modelo primeiro.")
    
    caminho_recente = max(arquivos, key=os.path.getctime)
    print(f"Carregando artefato: {caminho_recente}")
    
    # Carrega o dicionário salvo no train.py
    artifact = torch.load(caminho_recente, map_location=device)
    
    # Recria a arquitetura vazia e injeta os pesos treinados
    model = TabularModel(
        num_continuous_features=len(artifact['cont_cols']),
        categorical_cardinalities=artifact['cardinalities'],
        embed_dim=16,
        num_heads=4,
        mlp_hidden_dim=64,
        num_classes=2
    ).to(device)
    
    model.load_state_dict(artifact['model_state_dict'])
    model.eval() # Modo de inferência estrito
    print(f"Modelo carregado. Limiar Conformal de Segurança (q_hat): {artifact['q_hat']:.4f}")

@app.post("/predict")
def prever(dados: InferenciaRequest):
    try:
        # 1. Prepara as features contínuas aplicando o Scaler do Treinamento
        scalers = artifact['scalers']
        
        def normalizar(valor, nome_coluna):
            return (valor - scalers[nome_coluna]['mean']) / scalers[nome_coluna]['std']
        # Nota de MLOps: Num cenário 100% ideal, aplicaríamos o scaler de média/desvio padrão aqui.
        x_cont = [
            normalizar(dados.age, 'age'),
            normalizar(dados.fnlwgt, 'fnlwgt'),
            normalizar(dados.education_num, 'education-num'),
            normalizar(dados.capital_gain, 'capital-gain'),
            normalizar(dados.capital_loss, 'capital-loss'),
            normalizar(dados.hours_per_week, 'hours-per-week')
        ]
        tensor_cont = torch.tensor([x_cont], dtype=torch.float32).to(device)
        
        # 2. Transforma o texto das features categóricas em números usando o LabelEncoder salvo
        x_cat = []
        encoders = artifact['label_encoders']
        ordem_cat = artifact['cat_cols']
        
        for col in ordem_cat:
            valor_texto = getattr(dados, col.replace("-", "_")) # Ajusta nomes com hífen
            # Se o texto for algo que o modelo nunca viu, usamos a classe padrão (fallback)
            if valor_texto in encoders[col].classes_:
                valor_num = encoders[col].transform([valor_texto])[0]
            else:
                valor_num = 0 
            x_cat.append(valor_num)
            
        tensor_cat = torch.tensor([x_cat], dtype=torch.int64).to(device)
        
        # 3. Inferência Matemática
        with torch.no_grad():
            logits, attention_weights = model(tensor_cont, tensor_cat)
            probs = torch.softmax(logits, dim=1)
            prob_classe_1 = probs[0, 1].item()
            
            # Extraindo a primeira linha de pesos da atenção esparsa para explicabilidade
            pesos_esparsos = attention_weights[0, 0, 0, :].cpu().numpy().tolist()
            pesos_arredondados = [round(p, 4) for p in pesos_esparsos]

        # 4. Regra de Negócio Conformal (A IA sabe que não sabe?)
        q_hat = artifact['q_hat']
        margem_erro = 1.0 - prob_classe_1 if prob_classe_1 > 0.5 else prob_classe_1
        alerta_incerteza = bool(margem_erro > q_hat) # Se a margem for maior que o q_hat, aciona o alerta

        return {
            "status": "success",
            "prediction": {
                "probabilidade_fraude_ou_alto_risco": round(prob_classe_1, 4),
                "decisao_sugerida": "Intervenção Humana" if alerta_incerteza else ("Aprovar" if prob_classe_1 < 0.5 else "Negar"),
                "alerta_conformal_prediction": alerta_incerteza
            },
            "explicabilidade": {
                "pesos_atencao_esparsa": pesos_arredondados,
                "nota": "Valores 0.0 indicam que a IA ignorou a variável matematicamente."
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)