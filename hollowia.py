import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "corpus.txt"
MODELO_ARQ = BASE / "hollowia_modelo.pt"

# Modelo em nível de byte: cada valor UTF-8 é um token de 0 a 255.
VOCAB = 256
TAM_BLOCO = 256
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AtencaoMultiplaCabecas(nn.Module):
    def __init__(self, n_embd, n_head, tam_bloco):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd precisa ser divisível por n_head")
        self.n_head = n_head
        self.hd = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mascara", torch.tril(torch.ones(tam_bloco, tam_bloco)),
            persistent=False
        )

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_head, self.hd).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
        pesos = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        pesos = pesos.masked_fill(self.mascara[:t, :t] == 0, float("-inf"))
        pesos = F.softmax(pesos, dim=-1)
        saida = (pesos @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(saida)


class Bloco(nn.Module):
    def __init__(self, n_embd, n_head, tam_bloco):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.atencao = AtencaoMultiplaCabecas(n_embd, n_head, tam_bloco)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.atencao(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HollowIA(nn.Module):
    def __init__(
        self, tam_vocab=VOCAB, n_embd=N_EMBD, n_head=N_HEAD,
        n_layer=N_LAYER, tam_bloco=TAM_BLOCO
    ):
        super().__init__()
        self.tam_bloco = tam_bloco
        self.token_emb = nn.Embedding(tam_vocab, n_embd)
        self.pos_emb = nn.Embedding(tam_bloco, n_embd)
        self.blocos = nn.Sequential(
            *[Bloco(n_embd, n_head, tam_bloco) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.saida = nn.Linear(n_embd, tam_vocab)

    def forward(self, idx, alvos=None):
        _, t = idx.shape
        posicoes = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(posicoes)
        logits = self.saida(self.norm(self.blocos(x)))
        perda = None
        if alvos is not None:
            perda = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), alvos.reshape(-1)
            )
        return logits, perda

    @torch.no_grad()
    def gerar(self, idx, max_novos_tokens=240, temperatura=0.8, top_k=40):
        temperatura = max(float(temperatura), 0.05)
        estava_treinando = self.training
        self.eval()
        for _ in range(max_novos_tokens):
            contexto = idx[:, -self.tam_bloco:]
            logits, _ = self(contexto)
            logits = logits[:, -1, :] / temperatura
            if top_k and top_k < logits.size(-1):
                valores, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(
                    logits < valores[:, [-1]], float("-inf")
                )
            probabilidades = F.softmax(logits, dim=-1)
            proximo = torch.multinomial(probabilidades, 1)
            idx = torch.cat((idx, proximo), dim=1)
        self.train(estava_treinando)
        return idx


def carregar_dados():
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"Não encontrei {CORPUS.name}. Execute a coleta de textos primeiro."
        )
    texto = CORPUS.read_text(encoding="utf-8", errors="ignore")
    # Normalização leve: preserva acentos e pontuação, removendo excesso de espaços.
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    dados = texto.encode("utf-8")
    if len(dados) < TAM_BLOCO + 2:
        raise ValueError("O corpus é pequeno demais para o contexto configurado.")
    return torch.tensor(list(dados), dtype=torch.long)


def amostrar_lote(dados, tamanho):
    indices = torch.randint(0, len(dados) - TAM_BLOCO - 1, (tamanho,))
    x = torch.stack([dados[i:i + TAM_BLOCO] for i in indices])
    y = torch.stack([dados[i + 1:i + TAM_BLOCO + 1] for i in indices])
    return x.to(DEVICE), y.to(DEVICE)


def treinar(passo_max=1000, continuar=True):
    torch.manual_seed(42)
    dados = carregar_dados()
    modelo = HollowIA().to(DEVICE)

    if continuar and MODELO_ARQ.exists():
        try:
            estado = torch.load(MODELO_ARQ, map_location=DEVICE, weights_only=True)
            modelo.load_state_dict(estado)
            print("Continuando o treinamento do modelo existente.")
        except (RuntimeError, ValueError, TypeError, KeyError):
            print(
                "O modelo salvo usa outra arquitetura (por exemplo, contexto antigo). "
                "Começando um modelo novo."
            )

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=0.0003, weight_decay=0.01
    )
    modelo.train()
    print(
        f"Dispositivo: {DEVICE} | bytes no corpus: {len(dados):,} | "
        f"contexto: {TAM_BLOCO} | passos: {passo_max}"
    )

    for passo in range(1, max(1, passo_max) + 1):
        x, y = amostrar_lote(dados, BATCH_SIZE)
        _, perda = modelo(x, y)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()

        if passo == 1 or passo % 50 == 0 or passo == passo_max:
            print(
                f"Treino {passo}/{passo_max} | "
                f"perda: {perda.item():.4f} | dados: {len(dados):,} bytes"
            )

    temporario = MODELO_ARQ.with_suffix(".tmp")
    torch.save(modelo.state_dict(), temporario)
    temporario.replace(MODELO_ARQ)
    print(f"Modelo salvo em {MODELO_ARQ.name}")
    print("Dica: uma perda menor ajuda, mas não garante respostas coerentes.")


def carregar_modelo():
    if not MODELO_ARQ.exists():
        raise FileNotFoundError(
            "Ainda não existe um modelo treinado. Rode: python hollowia.py --treinar"
        )
    modelo = HollowIA().to(DEVICE)
    try:
        estado = torch.load(MODELO_ARQ, map_location=DEVICE, weights_only=True)
        modelo.load_state_dict(estado)
    except (RuntimeError, ValueError, TypeError, KeyError) as erro:
        raise RuntimeError(
            "O arquivo de pesos é incompatível com esta versão. "
            "Treine novamente com: python hollowia.py --treinar --novo"
        ) from erro
    modelo.eval()
    return modelo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento da HollowIA")
    parser.add_argument("--treinar", action="store_true")
    parser.add_argument("--passos", type=int, default=1000)
    parser.add_argument(
        "--novo", action="store_true",
        help="ignora os pesos existentes e começa o treinamento do zero"
    )
    args = parser.parse_args()

    if args.treinar:
        treinar(args.passos, continuar=not args.novo)
    else:
        print("Use --treinar para treinar a HollowIA.")
