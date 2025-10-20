from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
DADOS_JOGOS = DATA_DIR / "dados_jogos.csv"
MATRIZ = DATA_DIR / "matriz_utilidade.csv"

from collections import defaultdict
import numpy as np
import pandas as pd
import random
import re
import sys

def main():
    print("Iniciando o jogo...")
    jogos = carregar_dados(DADOS_JOGOS)
    matriz_utilidade = carregar_matriz(MATRIZ)
    print(f"Carregados {len(jogos)} jogos e {len(matriz_utilidade)} utilidades.")
    print("Iniciando o processo de seleção...")
    resultado = processar_jogos(jogos, matriz_utilidade)
    print("Resultado:", resultado)

def carregar_dados(caminho):
    dados = pd.read_csv(caminho)
    return dados

def carregar_matriz(caminho):
    matriz = pd.read_csv(caminho)
    return matriz

def processar_jogos(jogos, matriz_utilidade):
    # Cria um dicionário para armazenar os jogos e suas utilidades
    jogos_utilidades = defaultdict(list)
    for jogo in jogos:
        for utilidade in matriz_utilidade:
            jogos_utilidades[jogo].append(utilidade)
    # Seleciona o jogo com a maior utilidade
    jogo_selecionado = max(jogos_utilidades, key=lambda x: max(jogos_utilidades[x]))
    return jogo_selecionado

if __name__ == "__main__":
    main()