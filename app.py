# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="Sistema de Recomendação de Jogos", layout="wide")

# CSS para melhorar aparência dos botões de estrela
st.markdown("""
<style>
    .stButton > button {
        background: none;
        border: none;
        padding: 0;
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
    }
    .stButton > button:hover {
        color: gold;
    }
</style>
""", unsafe_allow_html=True)

# Adicione este CSS no início do arquivo, após st.set_page_config
st.markdown("""
<style>
    .game-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .rating-display {
        color: gold;
        font-size: 20px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def rating_stars(key, current_value=0):
    """Cria 5 botões de estrela para avaliação"""
    cols = st.columns(5)
    rating = current_value
    
    for i in range(5):
        with cols[i]:
            # Estrela preenchida se valor atual >= posição+1
            star = "⭐" if i < rating else "☆"
            if st.button(star, key=f"{key}_star_{i+1}"):
                rating = i + 1 if rating != i + 1 else 0
    
    return rating

# Configuração básica
st.set_page_config(page_title="Sistema de Recomendação de Jogos", layout="wide")

# Funções auxiliares
def safe_rerun():
    """Tenta recarregar a página de forma segura"""
    try:
        st.rerun()
    except:
        st.info("Por favor, atualize a página (F5)")
        st.stop()

def clean_email(email):
    """Limpa e normaliza o email"""
    return email.strip().lower() if email else ""

# Carregamento de dados
@st.cache_data
def load_data():
    try:
        df_jogos = pd.read_csv("dados_jogos.csv")
        df_matriz_utilidade = pd.read_csv("matriz_utilidade.csv", index_col=0)
        st.success("Dados carregados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        df_jogos = pd.DataFrame()
        df_matriz_utilidade = pd.DataFrame()
    return df_jogos, df_matriz_utilidade

# Inicialização do estado
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Carrega dados
df_jogos, df_matriz_utilidade = load_data()

# Interface principal
st.title("Sistema de Recomendação de Jogos")

# Página de login
if st.session_state.page == 'login':
    with st.form("login_form"):
        st.subheader("Login")
        email = st.text_input("E-mail")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
        
        if submitted:
            if not email.strip():
                st.error("Digite um e-mail válido")
            else:
                try:
                    clean_user_email = clean_email(email)
                    st.session_state.user_email = clean_user_email
                    st.session_state.page = 'rating'
                    
                    # Verifica/cria usuário na matriz
                    if clean_user_email not in df_matriz_utilidade.index:
                        nova_linha = pd.Series(0, index=df_jogos['nome_jogo'], name=clean_user_email)
                        df_matriz_utilidade = pd.concat([df_matriz_utilidade, nova_linha.to_frame().T])
                        df_matriz_utilidade.to_csv("matriz_utilidade.csv")
                        st.success("Novo usuário criado!")
                    
                    safe_rerun()
                except Exception as e:
                    st.error(f"Erro no login: {str(e)}")
                    st.session_state.page = 'login'
                    st.session_state.user_email = None
    
    st.info("Digite e-mail e senha para criar uma conta (protótipo)")

elif st.session_state.page == 'rating':
    try:
        # Header com logout
        col1, col2 = st.columns([3,1])
        with col2:
            st.write(f"Usuário: {st.session_state.user_email}")
            if st.button("Logout"):
                st.session_state.page = 'login'
                st.session_state.user_email = None
                st.session_state.user_ratings = {}
                safe_rerun()
        
        st.subheader("Avalie os Jogos")
        
        # Grid de cards (3 por linha)
        cols = st.columns(3)
        for i, row in df_jogos.iterrows():
            with cols[i % 3]:
                with st.container():
                    st.markdown('<div class="game-card">', unsafe_allow_html=True)
                    
                    # Imagem e informações do jogo
                    # st.image("https://placehold.in/200@2x", use_container_width=True)
                    st.markdown(f"**{row['nome_jogo']}**")
                    st.caption(f"{row['caracteristica_1']} | {row['caracteristica_2']}")
                    
                    # Slider de avaliação
                    try:
                        current_rating = df_matriz_utilidade.loc[st.session_state.user_email, row['nome_jogo']]
                        current_rating = int(current_rating) if pd.notnull(current_rating) else 0
                    except (KeyError, ValueError):
                        current_rating = 0
                    
                    rating = st.slider(
                        "Avaliação",
                        min_value=0,
                        max_value=5,
                        value=current_rating,
                        key=f"rating_{i}"
                    )
                    
                    # Exibe estrelas baseado na avaliação
                    if rating > 0:
                        st.markdown(
                            f'<div class="rating-display">{"★" * rating}{"☆" * (5-rating)}</div>',
                            unsafe_allow_html=True
                        )
                        st.session_state.user_ratings[row['nome_jogo']] = rating
                        df_matriz_utilidade.loc[st.session_state.user_email, row['nome_jogo']] = rating
                    else:
                        st.write("Sem avaliação")
                        st.session_state.user_ratings.pop(row['nome_jogo'], None)
                        df_matriz_utilidade.loc[st.session_state.user_email, row['nome_jogo']] = 0
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Salvar Avaliações"):
            df_matriz_utilidade.to_csv("matriz_utilidade.csv")
            st.success("Avaliações salvas!")
            
    except Exception as e:
        st.error(f"Erro ao carregar perfil: {str(e)}")
        st.session_state.page = 'login'
        safe_rerun()

elif st.session_state.page == 'recommendations':
    # Header com botões de navegação
    col1, col2 = st.columns([3,1])
    with col2:
        st.write(f"Usuário: {st.session_state.user_email}")
        if st.button("Voltar para Avaliações"):
            st.session_state.page = 'rating'
            safe_rerun()
        if st.button("Logout"):
            st.session_state.page = 'login'
            st.session_state.user_email = None
            st.session_state.user_ratings = {}
            safe_rerun()
    
    st.subheader("Suas Recomendações")
    
    recomendacoes = get_recommendations(st.session_state.user_email)
    if not recomendacoes:
        st.info("Avalie mais jogos para receber recomendações personalizadas!")
    else:
        for nome, score in recomendacoes:
            st.write(f"- {nome} (Relevância: {score:.2f})")