import pandas as pd
from pathlib import Path

p = Path(r"c:\Users\vinic\Documents\GitHub\IIA-Projeto1\matriz_utilidade.csv")
if not p.exists():
    print("Arquivo matriz_utilidade.csv não encontrado em:", p)
    raise SystemExit(1)

df = pd.read_csv(p, index_col=0)

# Normaliza índices (emails/nomes de usuário)
new_index = df.index.to_series().astype(str).str.strip().str.lower()
df.index = new_index

# Se houver duplicatas, mantemos a avaliação máxima por jogo (pode ajustar para média)
df = df.groupby(df.index).max()

# Salva versão limpa (faz backup do original)
backup = p.with_suffix(".backup.csv")
p.rename(backup)
df.to_csv(p)
print("matriz_utilidade.csv normalizada. Backup criado em:", backup)