import argparse
import hashlib
import random
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote_plus

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "corpus.txt"
URLS_ARQ = BASE / "urls_visitadas.txt"
USER_AGENT = "Mozilla/5.0 (compatible; HollowIAResearch/1.0)"
PALAVRAS_IGNORAR = set("de da do das dos e ou em no na nos nas um uma uns umas o os as que se por com para como ao aos foi sao são esta esse isso mais muito muita ja já quando onde qual quais sobre entre ate até porque mas nao não sim the and for with from this that have has was are you your into about after before using used".split())


def carregar_urls():
    try:
        return set(URLS_ARQ.read_text(encoding="utf-8").splitlines())
    except OSError:
        return set()


def salvar_url(url):
    with URLS_ARQ.open("a", encoding="utf-8") as arquivo:
        arquivo.write(url + "\n")


def extrair_pagina(url):
    try:
        resposta = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        resposta.raise_for_status()
        tipo = resposta.headers.get("content-type", "").lower()
        if "text/html" not in tipo and "text/plain" not in tipo:
            return "", []
        soup = BeautifulSoup(resposta.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
            tag.decompose()
        links = []
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"].strip()).split("#", 1)[0]
            parsed = urlparse(link)
            if parsed.scheme in ("http", "https") and parsed.netloc and not re.search(r"\.(pdf|jpg|jpeg|png|gif|zip|mp4|mp3|exe)(\?|$)", parsed.path, re.I):
                links.append(link)
        texto = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        return texto[:20000], links
    except requests.RequestException as erro:
        print(f"  página indisponível: {erro}")
        return "", []


def buscar_web(tema, max_resultados=8):
    """Busca sem duckduckgo-search/primp, incompatível com alguns Android/Python."""
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(tema)
    try:
        resposta = requests.get(
            url, timeout=20,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"},
        )
        resposta.raise_for_status()
        soup = BeautifulSoup(resposta.text, "html.parser")
        resultados, vistos = [], set()
        for a in soup.select("a.result__a, a.result-link"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            if "uddg=" in href:
                from urllib.parse import parse_qs
                href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
            href = href.split("#", 1)[0]
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or href in vistos:
                continue
            if "duckduckgo.com" in parsed.netloc:
                continue
            vistos.add(href)
            resultados.append({"href": href, "title": a.get_text(" ", strip=True)})
            if len(resultados) >= max_resultados:
                break
        if not resultados:
            print("  nenhum link encontrado; a busca pode estar bloqueada nesta conexão.")
        return resultados
    except requests.RequestException as erro:
        print(f"  busca falhou: {erro}")
        return []


def guardar_conhecimento(tema, url, texto):
    if len(texto) < 250:
        return False
    identificador = hashlib.sha256((url + texto[:500]).encode("utf-8", errors="ignore")).hexdigest()
    marcador = f"[FONTE:{identificador}]"
    try:
        with CORPUS.open("r", encoding="utf-8") as arquivo:
            if marcador in arquivo.read():
                return False
    except OSError:
        pass
    with CORPUS.open("a", encoding="utf-8") as arquivo:
        arquivo.write(f"\n\n=== TEMA: {tema} ===\nURL: {url}\n{marcador}\n{texto}\n")
    return True


def descobrir_temas(texto, visitados, quantidade=5):
    palavras = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9-]{4,}", texto.lower())
    frequentes = [p for p in palavras if p not in PALAVRAS_IGNORAR]
    temas = []
    for palavra in random.sample(frequentes, min(len(frequentes), 100)) if frequentes else []:
        if palavra not in visitados:
            visitados.add(palavra)
            temas.append(palavra)
            if len(temas) >= quantidade:
                break
    return temas


def pesquisar_tema(tema, visitadas, fila):
    print(f"\n[PESQUISA] {tema}")
    resultados = buscar_web(tema)
    random.shuffle(resultados)
    textos = []
    links_candidatos = []
    for resultado in resultados:
        url = resultado.get("href") or resultado.get("url")
        if not url or url in visitadas:
            continue
        visitadas.add(url)
        salvar_url(url)
        print(f"  lendo: {url}")
        texto, links = extrair_pagina(url)
        if texto:
            if guardar_conhecimento(tema, url, texto):
                print(f"  conhecimento salvo: {len(texto)} caracteres")
            textos.append(texto)
            links_candidatos.extend(links[:15])
    for link in links_candidatos:
        if link not in visitadas and len(fila) < 500:
            fila.append((tema, link))
    combinado = " ".join(textos)
    for novo_tema in descobrir_temas(combinado, visitadas, 5):
        fila.append((novo_tema, None))
    return combinado


def treinar_modelo(passos):
    print("\n[APRENDIZADO] Treinando a HollowIA com o corpus atualizado...")
    try:
        resultado = subprocess.run(
            [sys.executable, str(BASE / "hollowia.py"), "--treinar", "--passos", str(passos)],
            cwd=BASE, check=False
        )
        if resultado.returncode != 0:
            print("  treinamento não concluído; a pesquisa continuará.")
    except OSError as erro:
        print(f"  não foi possível iniciar o treinamento: {erro}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tema", nargs="+", help="tema inicial da pesquisa")
    parser.add_argument("--intervalo-treino", type=int, default=20)
    parser.add_argument("--passos-treino", type=int, default=250)
    parser.add_argument("--pausa", type=float, default=2.0)
    args = parser.parse_args()
    tema_inicial = " ".join(args.tema).strip()
    visitadas = carregar_urls()
    fila = deque([(tema_inicial, None)])
    temas_visitados = {tema_inicial.lower()}
    ciclo = 0
    print(f"HOLLOWIA PESQUISA CONTÍNUA — tema inicial: {tema_inicial}")
    print("Salvando fontes e textos em corpus.txt. Ctrl+C encerra e inicia um último treinamento.")
    try:
        while True:
            if not fila:
                tema = random.choice(list(temas_visitados))
                fila.append((tema, None))
            tema, url_direto = fila.popleft()
            ciclo += 1
            if url_direto:
                texto, links = extrair_pagina(url_direto)
                if texto and guardar_conhecimento(tema, url_direto, texto):
                    print(f"[PÁGINA] salvo {url_direto} ({len(texto)} caracteres)")
                for link in links:
                    if link not in visitadas and len(fila) < 500:
                        visitadas.add(link)
                        salvar_url(link)
                        fila.append((tema, link))
                for novo in descobrir_temas(texto, temas_visitados, 5):
                    fila.append((novo, None))
            else:
                temas_visitados.add(tema.lower())
                pesquisar_tema(tema, visitadas, fila)
            print(f"[CICLO {ciclo}] fila: {len(fila)} | URLs visitadas: {len(visitadas)} | temas: {len(temas_visitados)}")
            if args.intervalo_treino > 0 and ciclo % args.intervalo_treino == 0:
                treinar_modelo(args.passos_treino)
            time.sleep(max(0.0, args.pausa))
    except KeyboardInterrupt:
        print(f"\nPesquisa interrompida após {ciclo} ciclos.")
        treinar_modelo(args.passos_treino)
        print("Pesquisa salva em corpus.txt; modelo atualizado quando o treinamento foi concluído.")


if __name__ == "__main__":
    main()
