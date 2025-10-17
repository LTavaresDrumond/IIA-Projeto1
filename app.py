# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CARREGAMENTO DOS DADOS (O mesmo que você já tem) ---
dados_jogos = {
    'id_jogo': list(range(1, 23)),
    'nome_jogo': ['FIFA 24', 'Counter-Strike 2', 'League of Legends', 'Elden Ring', 'The Witcher 3', 'Stardew Valley', 'NBA 2K24', 'The Last of Us Part I','Batman: Arkham Asylum', "Marvel's Spider-Man", 'Street Fighter 6', 'Mortal Kombat 1','God of War Ragnarök', "Assassin's Creed Mirage", 'Top Eleven', 'Bomba Patch','Grand Theft Auto V', 'Minecraft', 'Roblox', 'Rocket League', 'Gran Turismo 7', 'Valorant'],
    'caracteristica_1': [ 'Esportes', 'FPS Tático', 'MOBA', 'RPG de Ação', 'RPG de Ação', 'Simulação', 'Esportes', 'Ação-Aventura','Ação-Aventura', 'Ação-Aventura', 'Luta', 'Luta','Ação-Aventura', 'Ação-Aventura', 'Esportes', 'Esportes','Ação-Aventura', 'Sandbox', 'Plataforma de Criação', 'Esportes','Corrida', 'FPS Tático' ],
    'caracteristica_2': [ 'Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online', 'Singleplayer', 'Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Singleplayer','Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Multiplayer Online','Singleplayer', 'Singleplayer', 'Multiplayer Online', 'Multiplayer Local','Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online', 'Multiplayer Online','Multiplayer Online', 'Multiplayer Online' ],
    'caracteristica_3': [ 'Competitivo', 'Competitivo', 'Estratégia', 'Mundo Aberto', 'Narrativa', 'Casual', 'Simulação Realista', 'Narrativa','Stealth', 'Mundo Aberto', 'Competitivo', 'Narrativa','Narrativa', 'Stealth', 'Gerenciamento', 'Modificação','Mundo Aberto', 'Criativo', 'Criativo', 'Arcade','Simulação Realista', 'Competitivo' ],
    'caracteristica_4': [ 'Futebol', 'Terrorismo', 'Fantasia', 'Fantasia Sombria','Fantasia Medieval', 'Vida Rural', 'Basquete', 'Pós-apocalíptico','Super-Herói', 'Super-Herói', 'Artes Marciais', 'Fantasia Sombria','Mitologia Nórdica', 'Histórico', 'Futebol', 'Futebol','Crime Moderno', 'Sobrevivência', 'User-Generated', 'Veicular','Automobilismo', 'Sci-Fi' ],
    'caracteristica_5': [ 'Realista', 'Realista', 'Estilizado', 'Realista','Realista', 'Pixel Art', 'Realista', 'Realista','Realista Sombrio', 'Realista', 'Estilizado', 'Realista','Realista', 'Realista', 'Interface 2D', 'Retrô','Realista', 'Voxel', 'Gráfico Simples', 'Cartunesco','Realista', 'Estilizado' ]
}
df_jogos = pd.DataFrame(dados_jogos)

# --- NOVA FUNÇÃO DE RECOMENDAÇÃO BASEADA EM PERFIL DE CARACTERÍSTICAS ---
def get_recommendations_from_profile(user_profile, df_jogos):
    # Função auxiliar para tratar características com espaços (ex: "FPS Tático" -> "fps_tático")
    def clean_text(text):
        return text.replace(' ', '_').lower()

    # Prepara uma cópia do DataFrame para não alterar o original
    df_jogos_cleaned = df_jogos.copy()
    
    # Limpa todas as características para que o TF-IDF as trate como palavras únicas
    for col in ['caracteristica_1', 'caracteristica_2', 'caracteristica_3', 'caracteristica_4', 'caracteristica_5']:
        df_jogos_cleaned[col] = df_jogos_cleaned[col].apply(clean_text)
    
    # Combina as características limpas em um único "documento" por jogo
    df_jogos_cleaned['perfil_conteudo'] = df_jogos_cleaned.iloc[:, 2:7].agg(' '.join, axis=1)

    # Vetorização com TF-IDF
    tfidf = TfidfVectorizer()
    matriz_tfidf_jogos = tfidf.fit_transform(df_jogos_cleaned['perfil_conteudo'])
    feature_names = tfidf.get_feature_names_out()
    feature_index_map = {feature: i for i, feature in enumerate(feature_names)}

    # Cria o vetor de perfil do usuário com base em suas escolhas
    perfil_usuario_vector = np.zeros(len(feature_names))

    # Atribui pesos positivos para características que o usuário gosta
    for char in user_profile.get('gosta', []):
        if clean_text(char) in feature_index_map:
            perfil_usuario_vector[feature_index_map[clean_text(char)]] = 5.0
    
    # Atribui pesos moderados para características que o usuário acha OK
    for char in user_profile.get('ok', []):
        if clean_text(char) in feature_index_map:
            perfil_usuario_vector[feature_index_map[clean_text(char)]] = 3.0

    # Atribui pesos NEGATIVOS para características que o usuário não gosta
    for char in user_profile.get('nao_gosta', []):
        if clean_text(char) in feature_index_map:
            perfil_usuario_vector[feature_index_map[clean_text(char)]] = -5.0

    # Calcula a similaridade entre o perfil do usuário e todos os jogos
    similaridades = cosine_similarity(perfil_usuario_vector.reshape(1, -1), matriz_tfidf_jogos)
    
    # Cria e retorna o DataFrame de recomendações
    df_recomendacoes = pd.DataFrame({
        'nome_jogo': df_jogos['nome_jogo'],
        'score': similaridades.flatten()
    }).sort_values(by='score', ascending=False)
    
    return df_recomendacoes

# --- LÓGICA DA INTERFACE ---

st.set_page_config(page_title="Sistema de Recomendação de Jogos", layout="centered")
st.title('🎮 Sistema de Recomendação de Jogos')

# Inicializa o estado da sessão para controlar as telas
if 'screen' not in st.session_state:
    st.session_state.screen = 'cadastro'
if 'profile' not in st.session_state:
    st.session_state.profile = {}

# --- TELA 1: CADASTRO ---
if st.session_state.screen == 'cadastro':
    st.header('Bem-vindo(a)!')
    nome_usuario = st.text_input('Para começar, por favor, insira seu nome:')
    
    if st.button('Próximo'):
        if nome_usuario:
            st.session_state.nome_usuario = nome_usuario
            st.session_state.screen = 'avaliacao'
            st.rerun()
        else:
            st.warning('Por favor, insira um nome para continuar.')

# --- TELA 2: CRIAÇÃO DE PERFIL POR CARACTERÍSTICAS (MODIFICADA) ---
elif st.session_state.screen == 'avaliacao':
    st.header(f'Olá, {st.session_state.nome_usuario}!')
    st.write('Para criar seu perfil, por favor, selecione as características que mais te atraem em um jogo.')

    # Pega todas as características únicas de todas as colunas
    caracteristicas_todas = set()
    for col in ['caracteristica_1', 'caracteristica_2', 'caracteristica_3', 'caracteristica_4', 'caracteristica_5']:
        caracteristicas_todas.update(df_jogos[col].unique())
    lista_caracteristicas = sorted(list(caracteristicas_todas))

    # Seleção de características que o usuário GOSTA
    gosta = st.multiselect(
        'Escolha 3 características que você **ADORA** em jogos:',
        lista_caracteristicas,
        max_selections=3,
        key='gosta_multiselect'
    )
    
    # Filtra as opções para não repetir
    opcoes_ok = [c for c in lista_caracteristicas if c not in gosta]
    ok = st.multiselect(
        'Escolha 3 características que você acha **OK**:',
        opcoes_ok,
        max_selections=3,
        key='ok_multiselect'
    )

    # Filtra as opções novamente
    opcoes_nao_gosta = [c for c in opcoes_ok if c not in ok]
    nao_gosta = st.multiselect(
        'Escolha 2 características que você **NÃO GOSTA** em jogos:',
        opcoes_nao_gosta,
        max_selections=2,
        key='nao_gosta_multiselect'
    )
        
    if st.button('Ver Recomendações'):
        # Validação para garantir que o usuário preencheu tudo
        if len(gosta) == 3 and len(ok) == 3 and len(nao_gosta) == 2:
            st.session_state.profile = {'gosta': gosta, 'ok': ok, 'nao_gosta': nao_gosta}
            st.session_state.screen = 'recomendacao'
            st.rerun()
        else:
            st.error('Por favor, selecione a quantidade exata de características em cada categoria (3, 3 e 2).')


# --- TELA 3: EXIBIÇÃO DAS RECOMENDAÇÕES ---
elif st.session_state.screen == 'recomendacao':
    st.header(f'Aqui estão suas recomendações, {st.session_state.nome_usuario}:')

    recomendacoes = get_recommendations_from_profile(st.session_state.profile, df_jogos)

    if recomendacoes.empty or recomendacoes['score'].max() <= 0:
        st.warning("Não foi possível gerar recomendações com base no seu perfil. Tente uma combinação diferente.")
    else:
        st.write("Com base no seu perfil, você provavelmente vai gostar destes jogos:")
        for index, row in recomendacoes.head(5).iterrows():
            st.subheader(f"{row['nome_jogo']}")
            st.write(f"_(Score de similaridade: {row['score']:.2f})_")
            caracteristicas = df_jogos[df_jogos['nome_jogo'] == row['nome_jogo']].iloc[0]
            st.info(f"**Gênero:** {caracteristicas['caracteristica_1']} | **Tema:** {caracteristicas['caracteristica_4']} | **Estilo Visual:** {caracteristicas['caracteristica_5']}")

    if st.button('Criar Novo Perfil'):
        # Limpa o estado da sessão para recomeçar
        keys_to_keep = ['nome_usuario'] # Mantém o nome do usuário
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state.screen = 'avaliacao' # Volta para a tela de avaliação
        st.rerun()