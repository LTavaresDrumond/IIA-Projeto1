import pandas as pd
import numpy as np

# Dicionário de 22 jogos com 5 caracteristicas cada
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

#Geração da matriz utilitária
# Número de usuários e jogos
num_usuarios = 500
num_jogos = len(df_jogos)

# Cria uma matriz cheia de zeros
matriz_utilidade = np.zeros((num_usuarios, num_jogos))

# Preenche a matriz com avaliações aleatórias
# Vamos fazer com que cada usuário avalie, em média, 5 jogos.
for i in range(num_usuarios):
    # Quantos jogos este usuário vai avaliar (entre 3 e 10, por exemplo)
    num_avaliacoes = np.random.randint(3, 11)
    
    # Escolhe aleatoriamente os jogos que serão avaliados
    jogos_avaliados_indices = np.random.choice(num_jogos, num_avaliacoes, replace=False)
    
    # Gera avaliações aleatórias para esses jogos (de 1 a 5)
    avaliacoes = np.random.randint(1, 6, size=num_avaliacoes)
    
    # Atribui as avaliações na matriz
    matriz_utilidade[i, jogos_avaliados_indices] = avaliacoes

# Converte para um DataFrame do Pandas para melhor visualização
colunas_jogos = df_jogos['nome_jogo'].tolist()
indices_usuarios = [f'user_{i+1}' for i in range(num_usuarios)]

df_matriz_utilidade = pd.DataFrame(matriz_utilidade, index=indices_usuarios, columns=colunas_jogos)

# Salva a matriz em um arquivo CSV para não precisar gerar toda vez
df_matriz_utilidade.to_csv('matriz_utilidade.csv')

print("Matriz de Utilidade Gerada:")
print(df_matriz_utilidade.head())