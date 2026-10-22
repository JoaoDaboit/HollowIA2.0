import json
import re
from pathlib import Path

import torch

from hollowia import CORPUS, DEVICE, carregar_modelo

BASE = Path(__file__).resolve().parent
MEMORIA_ARQ = BASE / "memoria.json"
MAX_MEMORIAS = 1000
MAX_TRECHOS = 2


def carregar_memoria():
    try:
        dados = json.loads(MEMORIA_ARQ.read_text(encoding="utf-8"))
        return dados if isinstance(dados, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def salvar_memoria(memoria):
    temporario = MEMORIA_ARQ.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(memoria[-MAX_MEMORIAS:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporario.replace(MEMORIA_ARQ)


def palavras(texto):
    """Tokenização simples em português para a busca BM25 local."""
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", texto.lower())


class IndiceBM25:
    """Implementação BM25 enxuta, sem dependências externas."""
    def __init__(self, documentos, k1=1.5, b=0.75):
        self.documentos = documentos
        self.k1 = k1
        self.b = b
        self.tokens = [palavras(doc) for doc in documentos]
        self.tamanhos = [len(doc) for doc in self.tokens]
        self.media = sum(self.tamanhos) / max(1, len(self.tamanhos))
        self.df = {}
        for tokens in self.tokens:
            for termo in set(tokens):
                self.df[termo] = self.df.get(termo, 0) + 1
        self.tf = []
        for tokens in self.tokens:
            contagem = {}
            for termo in tokens:
                contagem[termo] = contagem.get(termo, 0) + 1
            self.tf.append(contagem)

    def buscar(self, consulta, limite=MAX_TRECHOS):
        termos = set(palavras(consulta))
        if not termos or not self.documentos:
            return []
        total = len(self.documentos)
        pontuados = []
        for i, contagem in enumerate(self.tf):
            score = 0.0
            tamanho = self.tamanhos[i]
            for termo in termos:
                frequencia = contagem.get(termo, 0)
                if not frequencia:
                    continue
                documentos_com_termo = self.df.get(termo, 0)
                # IDF BM25 com suavização para manter pontuações positivas.
                idf = __import__("math").log(1 + (total - documentos_com_termo + 0.5) / (documentos_com_termo + 0.5))
                normalizador = frequencia + self.k1 * (1 - self.b + self.b * tamanho / max(1.0, self.media))
                score += idf * (frequencia * (self.k1 + 1)) / normalizador
            if score > 0:
                pontuados.append((score, self.documentos[i]))
        pontuados.sort(key=lambda item: item[0], reverse=True)
        resultado, vistos = [], set()
        for _, documento in pontuados:
            trecho = re.sub(r"\s+", " ", documento).strip()
            chave = trecho[:120]
            if len(trecho) < 80 or chave in vistos:
                continue
            vistos.add(chave)
            resultado.append(trecho[:700])
            if len(resultado) >= limite:
                break
        return resultado


def carregar_indice():
    """Lê o corpus e cria o índice BM25 uma vez ao iniciar a conversa."""
    if not CORPUS.exists():
        return IndiceBM25([])
    texto = CORPUS.read_text(encoding="utf-8", errors="ignore")
    blocos = re.split(r"(?<=[.!?])\s+|\n{2,}", texto)
    documentos = []
    for bloco in blocos:
        bloco = re.sub(r"\s+", " ", bloco).strip()
        if len(bloco) >= 80:
            documentos.append(bloco)
    return IndiceBM25(documentos)


_INDICE_BM25 = None

def buscar_contexto(pergunta, limite=MAX_TRECHOS):
    """Recupera trechos relevantes do corpus usando BM25 lexical (não semântico)."""
    global _INDICE_BM25
    if _INDICE_BM25 is None:
        _INDICE_BM25 = carregar_indice()
    return _INDICE_BM25.buscar(pergunta, limite)

def montar_prompt(pergunta, memoria, fontes, tam_bloco):
    historico = "\n".join(
        f"Usuário: {m.get('voce', '')}\nHollowIA: {m.get('hollowia', '')}"
        for m in memoria[-1:]
    )
    referencias = "\n".join(
        f"Trecho do corpus: {trecho}" for trecho in fontes
    )
    partes = [
        "Conversa em português.\n",
        historico,
        referencias,
        f"Usuário: {pergunta}\nHollowIA:",
    ]
    prompt = "\n".join(parte for parte in partes if parte)
    # Mantém o final do prompt, que contém a pergunta e o início da resposta.
    dados = prompt.encode("utf-8", errors="replace")[-(tam_bloco - 1):]
    return dados


def gerar_resposta(modelo, pergunta, memoria, fontes):
    prompt_bytes = montar_prompt(pergunta, memoria, fontes, modelo.tam_bloco)
    entrada = torch.tensor(
        [list(prompt_bytes)], dtype=torch.long, device=DEVICE
    )
    tamanho_prompt = entrada.shape[1]
    gerado = modelo.gerar(
        entrada, max_novos_tokens=240, temperatura=0.75, top_k=35
    )[0].tolist()

    novos = bytes(gerado[tamanho_prompt:])
    resposta = novos.decode("utf-8", errors="ignore").strip()

    # Para a geração não continuar escrevendo novos turnos ou referências.
    marcadores = ("\nUsuário:", "\nTrecho do corpus:", "\nConversa em português.")
    for marcador in marcadores:
        resposta = resposta.split(marcador)[0]
    resposta = resposta.strip()

    if not resposta:
        return (
            "Ainda não consegui formar uma resposta útil. "
            "Meu modelo ainda é experimental; tente treinar com mais texto limpo."
        )
    return resposta


def main():
    try:
        modelo = carregar_modelo()
    except (FileNotFoundError, RuntimeError, ValueError) as erro:
        print(f"Não consegui carregar a HollowIA: {erro}")
        print("Se o modelo for antigo, rode: python hollowia.py --treinar --novo")
        return

    memoria = carregar_memoria()
    print("\n=== HOLLOWIA — CONVERSA ===")
    print("Comandos: sair | esquecer\n")
    print("Busca RAG: BM25 local | modelo experimental; respostas podem não fazer sentido.\n")

    while True:
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté mais!")
            break

        if not entrada:
            continue
        comando = entrada.lower()

        if comando == "sair":
            break
        if comando == "esquecer":
            memoria = []
            salvar_memoria(memoria)
            print("HollowIA: memória da conversa apagada.\n")
            continue

        fontes = buscar_contexto(entrada)
        resposta = gerar_resposta(modelo, entrada, memoria, fontes)
        print(f"HollowIA: {resposta}\n")

        memoria.append({
            "voce": entrada,
            "hollowia": resposta,
            "fontes": fontes,
        })
        memoria = memoria[-MAX_MEMORIAS:]
        salvar_memoria(memoria)


if __name__ == "__main__":
    main()
