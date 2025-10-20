# app.py

import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def build_tfidf_for_games(df_jogos):
    docs = df_jogos[['caracteristica_1','caracteristica_2','caracteristica_3','caracteristica_4','caracteristica_5']].fillna('').agg(' '.join, axis=1).tolist()
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    return vec, X

def recommend_by_text(query: str, df_jogos, vec, X, top_n=5):
    if not query or query.strip() == "":
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = sims.argsort()[::-1][:top_n]
    return [(df_jogos.loc[i, 'nome_jogo'], float(sims[i])) for i in idx]

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
    """Carrega dados e garante que a matriz de utilidade tenha índices normalizados."""
    # dados_jogos
    if os.path.exists("dados_jogos.csv"):
        df_jogos = pd.read_csv("dados_jogos.csv")
    else:
        raise FileNotFoundError("dados_jogos.csv não encontrado. Gere ou copie o arquivo no diretório do app.")

    # matriz_utilidade
    if os.path.exists("matriz_utilidade.csv"):
        df_matriz = pd.read_csv("matriz_utilidade.csv", index_col=0)
        # normaliza índices (remove espaços, torna lower)
        df_matriz.index = df_matriz.index.to_series().astype(str).str.strip().str.lower()
        # se existirem duplicados após strip/lower, mantém avaliação máxima
        if df_matriz.index.duplicated().any():
            df_matriz = df_matriz.groupby(df_matriz.index).max()
        # garante colunas no mesmo order/nome dos jogos (preenche zeros se faltar)
        jogos = df_jogos['nome_jogo'].tolist()
        for j in jogos:
            if j not in df_matriz.columns:
                df_matriz[j] = 0.0
        # reindexa colunas na mesma ordem dos jogos
        df_matriz = df_matriz.reindex(columns=jogos, fill_value=0.0)
    else:
        # cria matriz vazia (0 usuários)
        jogos = df_jogos['nome_jogo'].tolist()
        df_matriz = pd.DataFrame(columns=jogos)
        # salva para persistência
        df_matriz.to_csv("matriz_utilidade.csv")

    return df_jogos, df_matriz

# Funções de recomendação (cacheadas para eficiência)
@st.cache_data
def calcular_similaridade_jogos(df_jogos):
    perfil = df_jogos[['caracteristica_1','caracteristica_2','caracteristica_3','caracteristica_4','caracteristica_5']].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(perfil)
    sim = cosine_similarity(matriz_tfidf)
    return sim

def get_recommendations(user_email, df_jogos, df_matriz, top_n=5):
    """Gera recomendações por conteúdo para user_email (email deve ser normalizado)."""
    if user_email is None:
        return []
    user = str(user_email).strip().lower()
    if user not in df_matriz.index:
        return []
    # pega avaliações > 0 do usuário
    row = df_matriz.loc[user]
    user_ratings = {col: int(v) for col, v in row.items() if pd.notnull(v) and float(v) > 0}
    if not user_ratings:
        return []
    sim = calcular_similaridade_jogos(df_jogos)
    name_to_idx = {name: idx for idx, name in enumerate(df_jogos['nome_jogo'])}
    scores = np.zeros(len(df_jogos))
    sim_sums = np.zeros(len(df_jogos))
    for nome, rating in user_ratings.items():
        if nome in name_to_idx:
            i = name_to_idx[nome]
            sim_col = sim[i]
            scores += sim_col * rating
            sim_sums += sim_col
    with np.errstate(divide='ignore', invalid='ignore'):
        pred = np.divide(scores, sim_sums)
        pred[np.isnan(pred)] = 0
    # bloquear já avaliados
    for nome in user_ratings:
        if nome in name_to_idx:
            pred[name_to_idx[nome]] = -1e9
    top_idx = np.argsort(-pred)[:top_n]
    return [(df_jogos.loc[i, 'nome_jogo'], float(pred[i])) for i in top_idx]

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
            # redireciona automaticamente para a página de Recomendações
            st.session_state.page = 'recommendations'
            try:
                st.rerun()
            except Exception:
                st.info("Atualize a página (F5) para ver as recomendações.")
                st.stop()

        # botão visível que leva à página de recomendações sem salvar
        if st.button("Ver Recomendações"):
            st.session_state.page = 'recommendations'
            try:
                st.rerun()
            except Exception:
                st.info("Atualize a página (F5) para ver as recomendações.")
                st.stop()

    except Exception as e:
        st.error(f"Erro ao carregar perfil: {str(e)}")
        st.session_state.page = 'login'
        safe_rerun()

# Página de Recomendações
if st.session_state.page == 'recommendations':
    st.header("Recomendações Personalizadas")
    if st.session_state.user_email is None:
        st.warning("Faça login e avalie alguns jogos antes de ver recomendações.")
    else:
        user = st.session_state.user_email.strip().lower()
        # recarrega a matriz para garantir persistência recente
        try:
            df_matriz_utilidade = pd.read_csv("matriz_utilidade.csv", index_col=0)
            df_matriz_utilidade.index = df_matriz_utilidade.index.to_series().astype(str).str.strip().str.lower()
            # garante colunas na ordem
            df_matriz_utilidade = df_matriz_utilidade.reindex(columns=df_jogos['nome_jogo'].tolist(), fill_value=0.0)
        except Exception:
            st.error("Erro ao carregar avaliações salvas.")
            st.stop()

        recomendacoes_top_n = st.slider("Número de recomendações", 1, 10, 5)
        recomendacoes = get_recommendations(user, df_jogos, df_matriz_utilidade, top_n=recomendacoes_top_n)

        if not recomendacoes:
            st.info("Não há recomendações — avalie pelo menos 3 jogos e salve as avaliações.")
        else:
            for nome, score in recomendacoes:
                st.markdown(f"**{nome}** — relevância: {score:.3f}")

        # campo de busca textual
        query = st.text_input("O que você está procurando? (ex.: 'mundo aberto fantasia narrativa')")
        col1, col2 = st.columns([3,1])
        with col2:
            search_btn = st.button("Buscar por Texto")
        # prepara tfidf se necessário
        vec, X = build_tfidf_for_games(df_jogos)
        if search_btn and query.strip():
            results = recommend_by_text(query, df_jogos, vec, X, top_n=10)
            if not results:
                st.info("Nenhum resultado para a consulta.")
            else:
                st.subheader("Resultados da busca por texto")
                for nome, score in results:
                    st.markdown(f"**{nome}** — similaridade: {score:.3f}")

        # opcional: combinar com recomendações por perfil (se quiser)
        combine = st.checkbox("Combinar com meu perfil (hybrid)", value=False)
        if combine:
            perfil_rec = get_recommendations(st.session_state.user_email, df_jogos, df_matriz_utilidade, top_n=10)
            # transforma em dicionário nome->score
            perfil_scores = {n: s for n, s in perfil_rec}
            # se query vazia, só mostra perfil; se tiver query, faz combinação simples
            if query.strip():
                text_results = dict(recommend_by_text(query, df_jogos, vec, X, top_n=len(df_jogos)))
                alpha = 0.6  # peso para texto
                combined = {}
                for i, row in df_jogos.iterrows():
                    name = row['nome_jogo']
                    s_text = text_results.get(name, 0.0)
                    s_perfil = perfil_scores.get(name, 0.0)
                    combined[name] = alpha * s_text + (1-alpha) * s_perfil
                top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]
                st.subheader("Resultados combinados (texto + perfil)")
                for nome, score in top:
                    st.markdown(f"**{nome}** — score combinado: {score:.3f}")

        col1, col2 = st.columns([3,1])
        with col2:
            if st.button("Voltar para Avaliações"):
                st.session_state.page = 'rating'
                try:
                    st.rerun()
                except Exception:
                    st.info("Atualize a página (F5).")
                    st.stop()