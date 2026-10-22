"""
Treinamento experimental aprimorado para a HollowIA.
Compatível com a versão de hollowia.py que define:
CORPUS, MODELO_ARQ, DEVICE, TAM_BLOCO, HollowIA.

Este script:
- separa uma parte do corpus para validação;
- mede perda de treino e validação;
- salva o melhor modelo de acordo com a validação;
- permite continuar de um checkpoint compatível;
- interrompe com Ctrl+C sem perder o melhor checkpoint.

Aviso: isso melhora a disciplina do treinamento, mas não garante fluência.
Um corpus de artigos da web não ensina, sozinho, a conversar como um assistente.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hollowia import CORPUS, MODELO_ARQ, DEVICE, TAM_BLOCO, HollowIA

BASE = Path(__file__).resolve().parent
MELHOR_ARQ = BASE / "hollowia_melhor.pt"
BATCH_SIZE = 8 if DEVICE == "cpu" else 16
AVALIACAO_LOTES = 12
TAXA_APRENDIZADO = 0.0002


def carregar_corpus():
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"Não encontrei {CORPUS.name}. Coloque o corpus na pasta do projeto."
        )
    texto = CORPUS.read_text(encoding="utf-8", errors="ignore")
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    # Compacta espaços horizontais sem destruir parágrafos.
    linhas = [" ".join(linha.split()) for linha in texto.split("\n")]
    texto = "\n".join(linha for linha in linhas if linha)
    dados = torch.tensor(list(texto.encode("utf-8")), dtype=torch.long)
    if len(dados) < TAM_BLOCO * 10:
        raise ValueError(
            "Corpus pequeno demais para esta configuração. Colete mais texto limpo."
        )
    return dados


def lote(dados, batch_size):
    indices = torch.randint(0, len(dados) - TAM_BLOCO - 1, (batch_size,))
    x = torch.stack([dados[i:i + TAM_BLOCO] for i in indices]).to(DEVICE)
    y = torch.stack([dados[i + 1:i + TAM_BLOCO + 1] for i in indices]).to(DEVICE)
    return x, y


@torch.no_grad()
def avaliar(modelo, treino, validacao):
    modelo.eval()
    resultados = {}
    for nome, dados in (("treino", treino), ("validacao", validacao)):
        perdas = []
        for _ in range(AVALIACAO_LOTES):
            x, y = lote(dados, BATCH_SIZE)
            _, perda = modelo(x, y)
            perdas.append(float(perda.item()))
        resultados[nome] = sum(perdas) / len(perdas)
    modelo.train()
    return resultados


def carregar_checkpoint(modelo, caminho):
    if not caminho.exists():
        return False
    try:
        estado = torch.load(caminho, map_location=DEVICE, weights_only=True)
        modelo.load_state_dict(estado)
        print(f"Checkpoint carregado: {caminho.name}")
        return True
    except (RuntimeError, ValueError, TypeError, KeyError) as erro:
        print(f"Checkpoint incompatível ({caminho.name}): {erro}")
        return False


def salvar_atomico(modelo, caminho):
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    torch.save(modelo.state_dict(), temporario)
    temporario.replace(caminho)


def treinar(passos, novo=False, avaliar_cada=100):
    torch.manual_seed(42)
    dados = carregar_corpus()
    corte = int(len(dados) * 0.9)
    treino = dados[:corte]
    validacao = dados[corte:]

    modelo = HollowIA().to(DEVICE)
    carregou = False
    if not novo:
        carregou = carregar_checkpoint(modelo, MELHOR_ARQ)
        if not carregou:
            carregou = carregar_checkpoint(modelo, MODELO_ARQ)
    if novo:
        print("Treinamento do zero solicitado.")

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=0.01
    )
    modelo.train()
    melhor_validacao = float("inf")
    sem_melhoria = 0

    print(f"Dispositivo: {DEVICE}")
    print(f"Bytes totais: {len(dados):,}")
    print(f"Treino: {len(treino):,} | validação: {len(validacao):,}")
    print(f"Contexto: {TAM_BLOCO} | passos: {passos}")
    if carregou:
        print("Os pesos existentes foram carregados; o otimizador recomeça.")

    try:
        for passo in range(1, max(1, passos) + 1):
            x, y = lote(treino, BATCH_SIZE)
            _, perda = modelo(x, y)
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()

            if passo == 1 or passo % avaliar_cada == 0 or passo == passos:
                metricas = avaliar(modelo, treino, validacao)
                print(
                    f"Passo {passo}/{passos} | "
                    f"perda treino: {metricas['treino']:.4f} | "
                    f"perda validação: {metricas['validacao']:.4f}"
                )

                if metricas["validacao"] < melhor_validacao:
                    melhor_validacao = metricas["validacao"]
                    sem_melhoria = 0
                    salvar_atomico(modelo, MELHOR_ARQ)
                    # Atualiza também o caminho que conversa.py espera.
                    salvar_atomico(modelo, MODELO_ARQ)
                    print("  -> Melhor checkpoint salvo.")
                else:
                    sem_melhoria += 1

                # Não é uma prova definitiva de overfitting; é um alerta simples.
                if sem_melhoria >= 8:
                    print(
                        "A validação não melhorou em 8 avaliações. "
                        "Parando para evitar continuar sem benefício claro."
                    )
                    break

    except KeyboardInterrupt:
        print("\nTreinamento interrompido. O melhor checkpoint já salvo foi preservado.")

    print(f"Melhor checkpoint: {MELHOR_ARQ.name}")
    print(f"Pesos para a conversa: {MODELO_ARQ.name}")
    print(
        "Importante: perda baixa mede previsão do próximo byte; "
        "não garante respostas corretas ou conversação natural."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passos", type=int, default=3000)
    parser.add_argument("--novo", action="store_true",
                        help="não carrega pesos antigos")
    parser.add_argument("--avaliar-cada", type=int, default=100)
    args = parser.parse_args()
    treinar(args.passos, novo=args.novo,
            avaliar_cada=max(10, args.avaliar_cada))
