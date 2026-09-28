import math
import torch
import torch.nn as nn
from typing import List, Tuple

class ContinuousTokenizer(nn.Module):
    """
    Projeta variáveis contínuas escalares em um espaço vetorial de dimensão d.
    """
    def __init__(self, num_continuous_features: int, embed_dim: int):
        super().__init__()
        # Pesos: [num_features, embed_dim]
        self.weight = nn.Parameter(torch.randn(num_continuous_features, embed_dim))
        # Viés: [num_features, embed_dim]
        self.bias = nn.Parameter(torch.zeros(num_continuous_features, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x tem shape: [batch_size, num_continuous_features]
        # Adicionamos uma dimensão extra para fazer o broadcast da multiplicação
        # Resultado shape: [batch_size, num_continuous_features, embed_dim]
        return x.unsqueeze(2) * self.weight + self.bias

class Sparsemax(nn.Module):
    """
    Projeção euclidiana no simplex de probabilidade.
    Substitui o Softmax para gerar distribuições de peso estritamente esparsas (com zeros exatos).
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Movemos a dimensão alvo para o final para facilitar as operações de álgebra linear
        original_size = z.size()
        z = z.transpose(self.dim, -1)
        d = z.size(-1)

        # Subtrai o valor máximo do tensor para impedir o colapso de precisão do float32
        z = z - z.max(dim=-1, keepdim=True)[0]

        # 1. Ordena os logits de forma decrescente
        z_sorted, _ = torch.sort(z, dim=-1, descending=True)
        
        # 2. Cria o vetor de índices (1, 2, ..., d)
        k_range = torch.arange(1, d + 1, device=z.device, dtype=z.dtype)
        
        # 3. Calcula a soma cumulativa para encontrar o limiar de corte (tau)
        z_cumsum = torch.cumsum(z_sorted, dim=-1)
        
        # 4. Condição matemática para encontrar o número de elementos não-nulos k(z)
        # 1 + k * z_{(k)} > sum_{j=1}^k z_{(j)}
        is_gt = (1.0 + k_range * z_sorted) > z_cumsum
        
        # Encontra o maior índice (k) onde a condição é verdadeira
        k = (is_gt * k_range).max(dim=-1, keepdim=True)[0]
        
        # 5. Calcula o limiar tau
        tau = (torch.gather(z_cumsum, -1, (k - 1).long()) - 1.0) / k
        
        # 6. Aplica a função ReLU (max(0, z - tau)) para zerar o resto
        p = torch.clamp(z - tau, min=0.0)
        
        return p.transpose(self.dim, -1)

class SparseTabularSelfAttention(nn.Module):
    """
    Auto-Atenção Multi-Head com mecanismo de seleção esparsa.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Substituímos o F.softmax por nossa nova camada matemática
        self.sparsemax = Sparsemax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, num_features, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Aplicação da esparsidade geométrica
        attention_weights = self.sparsemax(scores)
        
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, num_features, self.embed_dim)
        return self.out_proj(context), attention_weights

class CategoricalTokenizer(nn.Module):
    """
    Cria embeddings para variáveis categóricas, considerando diferentes cardinalidades.
    """
    def __init__(self, cardinalities: list, embed_dim: int):
        super().__init__()
        # ModuleList de Embeddings, um para cada coluna categórica
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=embed_dim) 
            for c in cardinalities
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x tem shape: [batch_size, num_categorical_features]
        # Iteramos sobre cada coluna e seu respectivo embedding
        embedded_features = [
            embedding_layer(x[:, i]) 
            for i, embedding_layer in enumerate(self.embeddings)
        ]
        # Empilhamos os tensores. Shape final: [batch_size, num_features, embed_dim]
        return torch.stack(embedded_features, dim=1)

class TabularModel(nn.Module):
    """
    Arquitetura de Deep Learning Tabular avançada combinando Tokenização, 
    Atenção Esparsa (Seleção de Features) e Classificação.
    """
    def __init__(
        self, 
        num_continuous_features: int, 
        categorical_cardinalities: List[int], 
        embed_dim: int = 16, 
        num_heads: int = 4,
        mlp_hidden_dim: int = 64,
        num_classes: int = 2
    ):
        super().__init__()
        
        # 1. Tokenizadores (Projeção para o Espaço Latente)
        self.cont_tokenizer = ContinuousTokenizer(num_continuous_features, embed_dim)
        self.cat_tokenizer = CategoricalTokenizer(categorical_cardinalities, embed_dim)
        
        # 2. Mecanismo de Atenção (Busca de Correlações Esparsas)
        self.attention = SparseTabularSelfAttention(embed_dim, num_heads)
        
        # 3. Cabeça de Classificação (MLP)
        # O tamanho da entrada do MLP será: (Total de Features) * embed_dim
        total_features = num_continuous_features + len(categorical_cardinalities)
        mlp_input_dim = total_features * embed_dim
        
        self.mlp_head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularização
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna os logits finais (predição) e a matriz de pesos de atenção para explicabilidade.
        """
        # Tokenização
        out_cont = self.cont_tokenizer(x_cont)
        out_cat = self.cat_tokenizer(x_cat)
        
        # Concatenação no eixo das features
        x_combined = torch.cat([out_cont, out_cat], dim=1)
        
        # Atenção Esparsa
        context_out, attention_weights = self.attention(x_combined)
        
        # Achatamento (Flatten) para a camada linear final
        batch_size = context_out.shape[0]
        context_flat = context_out.view(batch_size, -1)
        
        # Predição final
        logits = self.mlp_head(context_flat)
        
        return logits, attention_weights