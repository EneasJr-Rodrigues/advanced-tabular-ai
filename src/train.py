import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime

from data.dataset import AdvancedTabularDataset
from model.model import TabularModel
from model.loss import TabularFocalLoss
from sklearn.model_selection import train_test_split

def get_real_data():
    """Baixa e prepara o dataset real Adult Census (UCI) driblando o Kaggle."""
    print("Baixando dados reais da UCI Machine Learning...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    
    # engine='python' e sep=',\s+' removem os espaços em branco indesejados do CSV original
    df = pd.read_csv(url, names=columns, sep=',\s+', engine='python')
    df = df.dropna()

    # 1. Definindo e normalizando features contínuas
    cont_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    
    scalers = {}
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        scalers[col] = {'mean': mean, 'std': std} # Guardamos a régua!

    # 2. Definindo features categóricas e extraindo cardinalidade
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    cardinalities = []
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        cardinalities.append(len(le.classes_))

    # 3. Tratando o alvo
    df['target'] = (df['income'] == '>50K').astype(np.int64)
    
    return df, cont_cols, cat_cols, cardinalities, label_encoders, scalers

def calibracao_conformal(probs_calibracao: np.ndarray, labels_calibracao: np.ndarray, alpha: float = 0.10):
    """Calcula o limite de incerteza com 90% de garantia de cobertura."""
    scores_erro = np.where(labels_calibracao == 1, 1 - probs_calibracao, probs_calibracao)
    n = len(scores_erro)
    scores_ordenados = np.sort(scores_erro)
    indice_quantil = int(np.ceil((n + 1) * (1 - alpha)))
    indice_quantil = min(indice_quantil, n - 1)
    return scores_ordenados[indice_quantil]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Treinando no dispositivo: {device}")

    df, cont_cols, cat_cols, cardinalities, label_encoders, scalers = get_real_data()
    
    # DIVISÃO DE DADOS (Treino vs Teste/Calibração)
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['target'])
    print(f"Treino: {len(df_train)} registros | Teste: {len(df_test)} registros")
    
    train_dataset = AdvancedTabularDataset(df_train, cont_cols, cat_cols, target_col='target')
    test_dataset = AdvancedTabularDataset(df_test, cont_cols, cat_cols, target_col='target')
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = TabularModel(
        num_continuous_features=len(cont_cols),
        categorical_cardinalities=cardinalities,
        embed_dim=16,
        num_heads=4,
        mlp_hidden_dim=64,
        num_classes=2
    ).to(device)

    criterion = TabularFocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 5
    print("\n[ FASE 1: TREINAMENTO ]")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_cont, x_cat, labels in train_loader:
            x_cont, x_cat, labels = x_cont.to(device), x_cat.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x_cont, x_cat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}] | Loss Média: {(total_loss / len(train_loader)):.4f}")

    print("\n[ FASE 2: VALIDAÇÃO E CALIBRAÇÃO CONFORMAL ]")
    model.eval()
    todas_probs = []
    todos_labels = []
    
    with torch.no_grad():
        for x_cont, x_cat, labels in test_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            logits, attention_weights = model(x_cont, x_cat)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probabilidade da classe 1
            
            todas_probs.extend(probs.cpu().numpy())
            todos_labels.extend(labels.numpy())
            
    todas_probs = np.array(todas_probs)
    todos_labels = np.array(todos_labels)
    
    acertos = ((todas_probs > 0.5) == todos_labels).mean()
    print(f"Acurácia Bruta no Teste: {acertos:.2%}")
    
    q_hat = calibracao_conformal(todas_probs, todos_labels, alpha=0.10)
    print(f"Limite de Incerteza (q_hat): {q_hat:.4f}")
    print("Significado Prático: Se a incerteza do modelo for maior que este limite, a IA rejeita prever e pede análise humana.")
    
    # Explicabilidade: Verificando o Sparsemax na prática
    print("\n[ FASE 3: EXPLICABILIDADE ESPARSA ]")
    print("Pesos de atenção para o primeiro indivíduo do teste:")
    # Pegamos a atenção da primeira cabeça de foco, para a primeira feature, olhando o resto
    pesos_amostra = attention_weights[0, 0, 0, :].cpu().numpy() 
    print(np.round(pesos_amostra, 4))
    print("Observe os zeros exatos gerados pela nossa matemática geométrica!")
    print("\n[ FASE 4: EXPORTAÇÃO DE ARTEFATO (MLOps) ]")
    
    # Cria a pasta de artefatos na raiz do projeto
    os.makedirs("artifacts", exist_ok=True)
    
    # Cria uma tag de versão baseada no momento exato do treino (ex: 20260924_172700)
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = f"artifacts/tabular_model_{version_tag}.pt"
    
    # Empacota a "inteligência" matemática e as regras de tradução de dados
    artifact = {
        'model_state_dict': model.state_dict(),
        'cont_cols': cont_cols,
        'cat_cols': cat_cols,
        'cardinalities': cardinalities,
        'label_encoders': label_encoders,
        'scalers': scalers,  # A régua matemática vai junto para a nuvem
        'q_hat': q_hat
    }
    
    torch.save(artifact, artifact_path)
    print(f"Versão do modelo salva com sucesso em: {artifact_path}")

if __name__ == "__main__":
    main()