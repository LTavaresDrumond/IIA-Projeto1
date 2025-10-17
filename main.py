import pandas as pd

# Dicionário completo com a característica 5 atualizada para "Estilo Visual"
dados_jogos = {
    'id_jogo': list(range(1, 23)),
    'nome_jogo': [
        'FIFA 24', 'Counter-Strike 2', 'League of Legends', 'Elden Ring',
        'The Witcher 3', 'Stardew Valley', 'NBA 2K24', 'The Last of Us Part I',
        'Batman: Arkham Asylum', "Marvel's Spider-Man", 'Street Fighter 6', 'Mortal Kombat 1',
        'God of War Ragnarök', "Assassin's Creed Mirage", 'Top Eleven', 'Bomba Patch',
        'Grand Theft Auto V', 'Minecraft', 'Roblox', 'Rocket League',
        'Gran Turismo 7', 'Valorant'
    ],
    'caracteristica_1': [ # Gênero Principal
        'Esportes', 'FPS Tático', 'MOBA', 'RPG de Ação',
        'RPG de Ação', 'Simulação', 'Esportes', 'Ação-Aventura',
        'Ação-Aventura', 'Ação-Aventura', 'Luta', 'Luta',
        'Ação-Aventura', 'Ação-Aventura', 'Esportes', 'Esportes',
        'Ação-Aventura', 'Sandbox', 'Plataforma de Criação', 'Esportes',
        'Corrida', 'FPS Tático'
    ],
    'caracteristica_2': [ # Modo de Jogo
        'Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online', 'Singleplayer',
        'Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Singleplayer',
        'Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Multiplayer Online',
        'Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Multiplayer Local',
        'Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online',
        'Multiplayer Online', 'Multiplayer Online'
    ],
    'caracteristica_3': [ # Estilo/Tag
        'Competitivo', 'Competitivo', 'Estratégia', 'Mundo Aberto',
        'Narrativa', 'Casual', 'Simulação Realista', 'Narrativa',
        'Stealth', 'Mundo Aberto', 'Competitivo', 'Narrativa',
        'Narrativa', 'Stealth', 'Gerenciamento', 'Modificação',
        'Mundo Aberto', 'Criativo', 'Criativo', 'Arcade',
        'Simulação Realista', 'Competitivo'
    ],
    'caracteristica_4': [ # Tema
        'Futebol', 'Terrorismo', 'Fantasia', 'Fantasia Sombria',
        'Fantasia Medieval', 'Vida Rural', 'Basquete', 'Pós-apocalíptico',
        'Super-Herói', 'Super-Herói', 'Artes Marciais', 'Fantasia Sombria',
        'Mitologia Nórdica', 'Histórico', 'Futebol', 'Futebol',
        'Crime Moderno', 'Sobrevivência', 'User-Generated', 'Veicular',
        'Automobilismo', 'Sci-Fi'
    ],
    'caracteristica_5': [ # Estilo Visual
        'Realista', 'Realista', 'Estilizado', 'Realista',
        'Realista', 'Pixel Art', 'Realista', 'Realista',
        'Realista Sombrio', 'Realista', 'Estilizado', 'Realista',
        'Realista', 'Realista', 'Interface 2D', 'Retrô',
        'Realista', 'Voxel', 'Gráfico Simples', 'Cartunesco',
        'Realista', 'Estilizado'
    ]
}

# Criando o DataFrame com os dados
df_jogos = pd.DataFrame(dados_jogos)

# Imprimindo o DataFrame completo para visualização
print("DataFrame completo com a Característica 5 atualizada para 'Estilo Visual':")
print(df_jogos.to_string())

import numpy as np

# Número de usuários e jogos
num_usuarios = 500
num_jogos = len(df_jogos)

# Cria uma matriz cheia de zeros --> np.zeros cria uma matriz com 0s com as dimensões passadas
matriz_utilidade = np.zeros((num_usuarios, num_jogos))

# Preenche a matriz com avaliações aleatórias
# Vamos fazer com que cada usuário avalie, em média, 7 jogos.
for i in range(num_usuarios):
    # criar uma variável com Quantos jogos este usuário vai avaliar (entre 3 e 10, por exemplo)
    num_avaliacoes = np.random.randint(3, 11)

    # Escolhe aleatoriamente os jogos que serão avaliados
    # Criando um array com os indicies dos mesmos
    jogos_avaliados_indices = np.random.choice(num_jogos, num_avaliacoes, replace=False)

    # Cria um array com tamanho do numero de avaliações que Gera avaliações
    # aleatórias para esses jogos (de 1 a 5)
    avaliacoes = np.random.randint(1, 6, size=num_avaliacoes)

    # Garante que pelo menos uma avaliação seja 4 ou 5
    if np.all(avaliacoes < 4):
        # Esclhe uma avaliação a abaixo de 4 aleatoriamente para substituir por 4 ou 5
        id_a_ser_modificado = np.random.choice(len(avaliacoes))
        avaliacoes[id_a_ser_modificado] = np.random.randint(4, 6)


    # Atribui as avaliações na matriz utilidade acessando a linha i e substituindo 
    # As colunas pelas novas avaliações geradas aleatoriamente
    matriz_utilidade[i, jogos_avaliados_indices] = avaliacoes

# Converte para um DataFrame do Pandas para melhor visualização
colunas_jogos = df_jogos['nome_jogo'].tolist()
indices_usuarios = [f'user_{i+1}' for i in range(num_usuarios)]

df_matriz_utilidade = pd.DataFrame(matriz_utilidade, index=indices_usuarios, columns=colunas_jogos)

# Salva a matriz em um arquivo CSV para não precisar gerar toda vez
df_matriz_utilidade.to_csv('matriz_utilidade.csv')

print("Matriz de Utilidade Gerada:")
print(df_matriz_utilidade.head(5))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cria uma nova coluna no dataframe dos jogos e combina todas as características em uma única string por jogo
df_jogos['perfil_conteudo'] = df_jogos[['caracteristica_1', 'caracteristica_2', 'caracteristica_3', 'caracteristica_4', 'caracteristica_5']].agg(' '.join, axis=1)

# Aplicar o TF-IDF
# inicializa o objeto tfidfvectorizer
tfidf = TfidfVectorizer()#(stop_words='english')não sera necessário pois as características já estão apenas em palavras chaves

# fit_transform calcula o TF-IDF de cada palavra em da coluna "perfil_conteudo" e transforma em uma matriz
#"matriz_tfidf_jogos" onde linhas = jogos, e colunas = palavras unicas (das características dos jogos)
# e os valores dentro das matrizes são os scores TF-IDF que representam a importancia de cada característica para cada jogo 
matriz_tfidf_jogos = tfidf.fit_transform(df_jogos['perfil_conteudo'])

print("Forma da matriz TF-IDF (jogos x características únicas):", matriz_tfidf_jogos.shape)

# --- Lógica de recomendação para um usuário específico ---

id_usuario_exemplo = 'user_2'

# Pega as avaliações do usuário
avaliacoes_usuario = df_matriz_utilidade.loc[id_usuario_exemplo]

# Filtra jogos que he liked (score >= 4) and that he actually rated (> 0)
jogos_gostados = avaliacoes_usuario[(avaliacoes_usuario >= 4) & (avaliacoes_usuario > 0)]
indices_jogos_gostados = [colunas_jogos.index(jogo) for jogo in jogos_gostados.index]

# Check if the user has rated any games highly
if not indices_jogos_gostados:
    print(f"Usuário {id_usuario_exemplo} não avaliou nenhum jogo com nota 4 ou superior.")
    print("Recomendando jogos populares em vez disso:")

    # Recommend popular games if no highly-rated games are found
    # Calculate popularity based on the number of ratings for each game
    popularidade_jogos = (df_matriz_utilidade > 0).sum(axis=0).sort_values(ascending=False)
    recomendacoes_populares = popularidade_jogos.head(5).index.tolist()

    df_recomendacoes = pd.DataFrame({
        'jogo': recomendacoes_populares,
        'score': [0] * len(recomendacoes_populares) # Assign a dummy score
    })

else:
    # Cria o perfil do usuário: média dos vetores TF-IDF dos jogos que he liked
    vetores_jogos_gostados = matriz_tfidf_jogos[indices_jogos_gostados]
    perfil_usuario = vetores_jogos_gostados.mean(axis=0).A1 # Convert to numpy array

    # Calcula a similaridade do perfil do usuário com todos os jogos
    similaridades = cosine_similarity(perfil_usuario.reshape(1, -1), matriz_tfidf_jogos)

    # Obtém uma lista de scores de similaridade
    scores_similaridade = similaridades.flatten()

    # Recomenda os jogos com maior similaridade (that the user hasn't rated yet)
    jogos_avaliados_pelo_usuario = avaliacoes_usuario[avaliacoes_usuario > 0].index
    df_recomendacoes = pd.DataFrame({
        'jogo': colunas_jogos,
        'score': scores_similaridade
    })

    # Filtra os já avaliados
    df_recomendacoes = df_recomendacoes[~df_recomendacoes['jogo'].isin(jogos_avaliados_pelo_usuario)]

    # Ordena pelos melhores scores
    df_recomendacoes = df_recomendacoes.sort_values(by='score', ascending=False)


print(f"\nRecomendações para {id_usuario_exemplo}:")
print(df_recomendacoes.head(5))