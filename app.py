# app.py

import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PT_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "à",
    "às",
    "o",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "sem",
    "e",
    "é",
    "ou",
    "que",
    "como",
    "se",
    "são",
    "mais",
    "mas",
    "foi",
    "ser",
    "tem",
    "há",
    "pelo",
    "pela",
    "pelos",
    "pelas",
    "até",
}
DEFAULT_STOPWORDS = list(set(ENGLISH_STOP_WORDS).union(PT_STOPWORDS))


def build_weighted_documents(df_jogos, weight_map=None):
    """
    Constrói documentos textuais por jogo aplicando pesos por campo.
    weight_map: dict campo->int (repetições) ex: {'caracteristica_1':3, 'caracteristica_3':2}
    """
    if weight_map is None:
        weight_map = {}
    fields = [
        "caracteristica_1",
        "caracteristica_2",
        "caracteristica_3",
        "caracteristica_4",
        "caracteristica_5",
    ]
    docs = []
    for _, row in df_jogos.iterrows():
        parts = []
        for f in fields:
            text = str(row.get(f, "")).strip()
            if not text:
                continue
            reps = int(weight_map.get(f, 1))
            parts.extend([text] * reps)
        docs.append(" ".join(parts))
    return docs


@st.cache_data
def build_tfidf_for_games(
    df_jogos,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    weight_map=None,
    stop_words=None,
):
    """
    Constrói vetorizador TF-IDF para os jogos.
    Retorna: (vectorizer, X_tfidf)
    """
    if stop_words is None:
        stop_words = DEFAULT_STOPWORDS  # list, compatível com sklearn
    docs = build_weighted_documents(df_jogos, weight_map=weight_map)
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )
    X = vec.fit_transform(docs)
    return vec, X


def recommend_by_text(query: str, df_jogos=None, vec=None, X=None, top_n=5):
    """mantém compatibilidade — aceita query, devolve top_n (nome, score)."""
    if not query or query.strip() == "":
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = sims.argsort()[::-1][:top_n]
    return [(df_jogos.loc[i, "nome_jogo"], float(sims[i])) for i in idx]


def recommend_for_user_from_train(
    user: str, df_jogos: pd.DataFrame, train_df: pd.DataFrame, X, top_n: int = 10
) -> List[str]:
    """Gera top-N recomendações por conteúdo dado X (TF-IDF) e train_df com avaliações persistidas."""
    if user not in train_df.index:
        return []
    row = train_df.loc[user]
    user_ratings = {
        col: int(v) for col, v in row.items() if pd.notnull(v) and float(v) > 0
    }
    if not user_ratings:
        return []
    name_to_idx = {name: idx for idx, name in enumerate(df_jogos["nome_jogo"])}
    sim = cosine_similarity(X)
    scores = np.zeros(len(df_jogos))
    sim_sums = np.zeros(len(df_jogos))
    for nome, rating in user_ratings.items():
        if nome in name_to_idx:
            i = name_to_idx[nome]
            sim_col = sim[i]
            scores += sim_col * rating
            sim_sums += sim_col
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.divide(scores, sim_sums)
        pred[np.isnan(pred)] = 0
    for nome in user_ratings:
        if nome in name_to_idx:
            pred[name_to_idx[nome]] = -1e9
    top_idx = np.argsort(-pred)[:top_n]
    return [df_jogos.loc[i, "nome_jogo"] for i in top_idx]


def precision_at_k(recommended: List[str], ground_truth: set, k: int) -> float:
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in ground_truth)
    return hits / k


def recall_at_k(recommended: List[str], ground_truth: set, k: int) -> float:
    if not ground_truth:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in ground_truth)
    return hits / len(ground_truth)


def apk(recommended: List[str], ground_truth: set, k: int) -> float:
    if not ground_truth:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(recommended[:k], start=1):
        if r in ground_truth:
            hits += 1
            score += hits / i
    return score / min(len(ground_truth), k)


def evaluate_offline_streamlit(
    df_jogos: pd.DataFrame,
    df_matriz: pd.DataFrame,
    ngram_range=(1, 2),
    min_df=1,
    weight_map=None,
    min_ratings=5,
    n_test=1,
    top_k=10,
    sample_users: int = 0,
) -> Dict[str, float]:
    """
    Executa avaliação leave-one-out (n_test por usuário) e retorna métricas agregadas.
    sample_users: 0 = usar todos; >0 = tamanho da amostra aleatória de usuários elegíveis.
    """
    # prepara dados
    jogos = df_jogos["nome_jogo"].tolist()
    df_matriz = df_matriz.reindex(columns=jogos, fill_value=0.0)
    user_counts = (df_matriz > 0).sum(axis=1)
    users = user_counts[user_counts >= min_ratings].index.tolist()
    if sample_users and sample_users < len(users):
        np.random.seed(42)
        users = list(np.random.choice(users, sample_users, replace=False))
    if not users:
        return {"precision": 0.0, "recall": 0.0, "map": 0.0, "n_users": 0}

    # TF-IDF com os parâmetros escolhidos
    vec, X = build_tfidf_for_games(
        df_jogos, ngram_range=ngram_range, min_df=min_df, weight_map=weight_map
    )

    precisions, recalls, apks = [], [], []
    progress = 0
    start = time.time()
    for u in users:
        # selecciona itens e faz holdout
        items = df_matriz.columns[(df_matriz.loc[u] > 0).values].tolist()
        if len(items) <= n_test:
            continue
        test_items = list(np.random.choice(items, n_test, replace=False))
        train_df = df_matriz.copy()
        for t in test_items:
            train_df.at[u, t] = 0.0
        recs = recommend_for_user_from_train(u, df_jogos, train_df, X, top_n=top_k)
        precisions.append(precision_at_k(recs, set(test_items), top_k))
        recalls.append(recall_at_k(recs, set(test_items), top_k))
        apks.append(apk(recs, set(test_items), top_k))
        progress += 1
    elapsed = time.time() - start
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "map": float(np.mean(apks)) if apks else 0.0,
        "n_users": progress,
        "time_sec": elapsed,
    }


# Configuração da página
st.set_page_config(
    page_title="NextGame - Recomendações Inteligentes de Jogos", layout="wide"
)

# CSS para melhorar aparência geral
st.markdown(
    """
<style>
    /* Title styling */
    .main h1 {
        text-align: center;
    }

    .main > div:first-child > div > div > div > p {
        text-align: center;
    }

    /* Remove anchor link buttons from headers - all possible selectors */
    h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
        display: none !important;
    }

    [data-testid="stHeader"] a {
        display: none !important;
    }

    /* Button styling - white background on hover */
    .stButton > button {
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    /* Star rating buttons */
    .stButton.star-button > button {
        background: none;
        border: none;
        padding: 0;
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
    }
    .stButton.star-button > button:hover {
        color: gold !important;
        background: none !important;
        border: none !important;
    }

    /* Game card styling - apply to containers in columns */
    [data-testid="column"] > div > div > div[data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        min-height: 600px;
        display: flex;
        flex-direction: column;
    }

    /* Ensure images maintain aspect ratio and size */
    [data-testid="stImage"] {
        width: 100%;
        min-height: 352px;
    }
    
    [data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        min-height: 352px !important;
        object-fit: cover !important;
        display: block !important;
    }

    /* Rating display */
    .rating-display {
        color: gold;
        font-size: 20px;
        margin-top: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)


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
    except Exception:
        st.info("Por favor, atualize a página (F5)")
        st.stop()


def clean_email(email):
    """Limpa e normaliza o email"""
    return email.strip().lower() if email else ""


# IGDB API Functions
@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_igdb_access_token(client_id: str, client_secret: str) -> str:
    """Obtém token de acesso da IGDB API"""
    try:
        url = "https://id.twitch.tv/oauth2/token"
        params = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
        response = requests.post(url, params=params)
        if response.status_code == 200:
            return response.json().get("access_token", "")
        return ""
    except Exception:
        return ""


@st.cache_data(ttl=604800)  # Cache por 7 dias
def search_game_image(game_name: str, client_id: str, access_token: str) -> str:
    """Busca imagem do jogo na IGDB API"""
    # Fallback: imagem padrão (ícone de jogo genérico)
    fallback_image = "https://placehold.co/264x352/1a1a2e/white?text=Game"
    
    if not client_id or not access_token:
        return fallback_image

    try:
        url = "https://api.igdb.com/v4/games"
        headers = {"Client-ID": client_id, "Authorization": f"Bearer {access_token}"}

        # Busca o jogo pelo nome
        data = f'search "{game_name}"; fields name,cover.url; limit 1;'
        response = requests.post(url, headers=headers, data=data, timeout=5)

        if response.status_code == 200:
            games = response.json()
            if games and len(games) > 0:
                game = games[0]
                if "cover" in game and "url" in game["cover"]:
                    # Converte para imagem grande (264x352)
                    cover_url = game["cover"]["url"]
                    cover_url = cover_url.replace("t_thumb", "t_cover_big")
                    return "https:" + cover_url

        # Se não encontrou a imagem, retorna fallback
        return fallback_image
    except Exception:
        return fallback_image


def get_game_image(game_name: str) -> str:
    """Obtém imagem do jogo com fallback para placeholder"""
    fallback_image = "https://placehold.co/264x352/1a1a2e/white?text=Game"
    
    try:
        # Configurações da API - você precisa obter estas credenciais em https://dev.twitch.tv/console
        # Para desenvolvimento, pode deixar vazio e usar placeholders
        IGDB_CLIENT_ID = st.secrets.get("IGDB_CLIENT_ID", "")
        IGDB_CLIENT_SECRET = st.secrets.get("IGDB_CLIENT_SECRET", "")
    except Exception:
        # Se não houver secrets configurados, usa placeholder
        return fallback_image

    if not IGDB_CLIENT_ID or not IGDB_CLIENT_SECRET:
        # Usa placeholder se não tiver credenciais configuradas
        return fallback_image

    # Obtém token de acesso
    access_token = get_igdb_access_token(IGDB_CLIENT_ID, IGDB_CLIENT_SECRET)

    if not access_token:
        return fallback_image

    # Busca imagem do jogo
    return search_game_image(game_name, IGDB_CLIENT_ID, access_token)


# caminho relativo do projeto (pasta onde está app.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  # se quiser usar subpasta, por ex. BASE_DIR / "data"


# Carregamento de dados
@st.cache_data
def load_data():
    """Carrega dados e garante que a matriz de utilidade tenha índices normalizados."""
    dados_path = DATA_DIR / "dados_jogos.csv"
    matriz_path = DATA_DIR / "matriz_utilidade.csv"

    # dados_jogos
    if dados_path.exists():
        df_jogos = pd.read_csv(dados_path)
    else:
        raise FileNotFoundError(
            f"{dados_path} não encontrado. Gere ou copie o arquivo no diretório do app."
        )

    # matriz_utilidade
    if matriz_path.exists():
        df_matriz = pd.read_csv(matriz_path, index_col=0)
        # normaliza índices (remove espaços, torna lower)
        df_matriz.index = (
            df_matriz.index.to_series().astype(str).str.strip().str.lower()
        )
        # se existirem duplicados após strip/lower, mantém avaliação máxima
        if df_matriz.index.duplicated().any():
            df_matriz = df_matriz.groupby(df_matriz.index).max()
        # garante colunas no mesmo order/nome dos jogos (preenche zeros se faltar)
        jogos = df_jogos["nome_jogo"].tolist()
        for j in jogos:
            if j not in df_matriz.columns:
                df_matriz[j] = 0.0
        # reindexa colunas na mesma ordem dos jogos
        df_matriz = df_matriz.reindex(columns=jogos, fill_value=0.0)
    else:
        # cria matriz vazia (0 usuários)
        jogos = df_jogos["nome_jogo"].tolist()
        df_matriz = pd.DataFrame(columns=jogos)
        # salva para persistência
        df_matriz.to_csv(matriz_path)

    return df_jogos, df_matriz


# Funções de recomendação (cacheadas para eficiência)
@st.cache_data
def calcular_similaridade_jogos(df_jogos):
    perfil = (
        df_jogos[
            [
                "caracteristica_1",
                "caracteristica_2",
                "caracteristica_3",
                "caracteristica_4",
                "caracteristica_5",
            ]
        ]
        .fillna("")
        .agg(" ".join, axis=1)
    )
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
    user_ratings = {
        col: int(v) for col, v in row.items() if pd.notnull(v) and float(v) > 0
    }
    if not user_ratings:
        return []
    sim = calcular_similaridade_jogos(df_jogos)
    name_to_idx = {name: idx for idx, name in enumerate(df_jogos["nome_jogo"])}
    scores = np.zeros(len(df_jogos))
    sim_sums = np.zeros(len(df_jogos))
    for nome, rating in user_ratings.items():
        if nome in name_to_idx:
            i = name_to_idx[nome]
            sim_col = sim[i]
            scores += sim_col * rating
            sim_sums += sim_col
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.divide(scores, sim_sums)
        pred[np.isnan(pred)] = 0
    # bloquear já avaliados
    for nome in user_ratings:
        if nome in name_to_idx:
            pred[name_to_idx[nome]] = -1e9
    top_idx = np.argsort(-pred)[:top_n]
    return [(df_jogos.loc[i, "nome_jogo"], float(pred[i])) for i in top_idx]


# Inicialização do estado
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "registered_users" not in st.session_state:
    st.session_state.registered_users = {}
if "users_loaded" not in st.session_state:
    st.session_state.users_loaded = False

# Carrega dados
df_jogos, df_matriz_utilidade = load_data()


def save_ratings_to_file(user_email, ratings_dict):
    """Salva as avaliações de um usuário no arquivo CSV"""
    # Recarrega a matriz do arquivo para ter os dados mais recentes
    matriz_path = DATA_DIR / "matriz_utilidade.csv"
    if matriz_path.exists():
        df_temp = pd.read_csv(matriz_path, index_col=0)
        df_temp.index = df_temp.index.to_series().astype(str).str.strip().str.lower()
    else:
        df_temp = pd.DataFrame(columns=df_jogos["nome_jogo"])

    # Adiciona usuário se não existir
    if user_email not in df_temp.index:
        nova_linha = pd.Series(0, index=df_jogos["nome_jogo"], name=user_email)
        df_temp = pd.concat([df_temp, nova_linha.to_frame().T])

    # Atualiza as avaliações
    for jogo, rating in ratings_dict.items():
        if jogo in df_temp.columns:
            df_temp.loc[user_email, jogo] = rating

    # Salva no arquivo
    df_temp.to_csv(matriz_path)
    return df_temp


def save_users():
    """Save registered users to file"""
    users_df = pd.DataFrame(
        [
            {"email": email, "password": password}
            for email, password in st.session_state.registered_users.items()
        ]
    )
    users_df.to_csv(DATA_DIR / "users.csv", index=False)


# Load registered users from a simple file (only once per session)
if not st.session_state.users_loaded:
    users_file = DATA_DIR / "users.csv"
    if users_file.exists():
        try:
            users_df = pd.read_csv(
                users_file, dtype=str
            )  # Force all columns to be strings
            # Create dictionary from email and password columns
            st.session_state.registered_users = {
                str(row["email"]): str(row["password"])
                for _, row in users_df.iterrows()
            }
        except Exception:
            st.session_state.registered_users = {}
    st.session_state.users_loaded = True


# Interface principal - Title
st.write(
    "<div style='text-align: center;'><span style='font-size: 80px;'>🎮</span></div>",
    unsafe_allow_html=True,
)
st.write(
    "<h1 style='text-align: center; margin-top: -20px;'>NextGame</h1>",
    unsafe_allow_html=True,
)
st.write(
    "<p style='text-align: center; color: gray; margin-top: -10px;'>Seu Sistema Inteligente de Recomendação de Jogos</p>",
    unsafe_allow_html=True,
)

# Página de login
if st.session_state.page == "login":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("### Bem-vindo de volta!")

        # Using text inputs outside form for Enter key support
        email = st.text_input("E-mail", key="login_email")
        password = st.text_input("Senha", type="password", key="login_password")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            login_clicked = st.button(
                "Entrar", type="primary", use_container_width=True
            )
        with col_btn2:
            register_clicked = st.button("Registrar", use_container_width=True)

        # Handle Enter key - check if both fields are filled
        if email and password and not login_clicked and not register_clicked:
            # This creates implicit "press enter to login" behavior
            login_clicked = True

        if login_clicked:
            if not email.strip():
                st.error("Digite um e-mail válido")
            elif not password.strip():
                st.error("Digite uma senha")
            else:
                clean_user_email = clean_email(email)

                # Check if user exists and password matches
                if clean_user_email in st.session_state.registered_users:
                    stored_password = st.session_state.registered_users[
                        clean_user_email
                    ]

                    if stored_password == password:
                        st.session_state.user_email = clean_user_email
                        st.session_state.page = "rating"
                        safe_rerun()
                    else:
                        st.error("Senha incorreta")
                else:
                    st.error("Usuário não encontrado. Por favor, registre-se primeiro.")

        if register_clicked:
            st.session_state.page = "register"
            safe_rerun()

# Página de registro
elif st.session_state.page == "register":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("### Criar Sua Conta")

        email = st.text_input("E-mail", key="register_email")
        password = st.text_input("Senha", type="password", key="register_password")
        password_confirm = st.text_input(
            "Confirmar Senha", type="password", key="register_password_confirm"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            register_submit = st.button(
                "Criar Conta", type="primary", use_container_width=True
            )
        with col_btn2:
            back_to_login = st.button("Voltar ao Login", use_container_width=True)

        # Handle Enter key for registration
        if (
            email
            and password
            and password_confirm
            and not register_submit
            and not back_to_login
        ):
            register_submit = True

        if register_submit:
            if not email.strip():
                st.error("Digite um e-mail válido")
            elif not password.strip():
                st.error("Digite uma senha")
            elif password != password_confirm:
                st.error("As senhas não coincidem")
            else:
                clean_user_email = clean_email(email)

                # Check if user already exists
                if clean_user_email in st.session_state.registered_users:
                    st.error("Este e-mail já está registrado. Faça login.")
                else:
                    try:
                        # Register user
                        st.session_state.registered_users[clean_user_email] = password
                        save_users()

                        # Create user in matrix
                        if clean_user_email not in df_matriz_utilidade.index:
                            nova_linha = pd.Series(
                                0, index=df_jogos["nome_jogo"], name=clean_user_email
                            )
                            df_matriz_utilidade = pd.concat(
                                [df_matriz_utilidade, nova_linha.to_frame().T]
                            )
                            df_matriz_utilidade.to_csv("matriz_utilidade.csv")

                        st.success(
                            "Conta criada com sucesso! Redirecionando para login..."
                        )
                        st.session_state.page = "login"
                        time.sleep(1)
                        safe_rerun()
                    except Exception as e:
                        st.error(f"Erro ao criar conta: {str(e)}")

        if back_to_login:
            st.session_state.page = "login"
            safe_rerun()

elif st.session_state.page == "rating":
    try:
        # Header com logout
        col1, col2 = st.columns([3, 1])
        with col2:
            st.write(f"Usuário: {st.session_state.user_email}")
            if st.button("Logout"):
                st.session_state.page = "login"
                st.session_state.user_email = None
                st.session_state.user_ratings = {}
                safe_rerun()

        st.subheader("Avalie os Jogos")

        # Grid de cards (3 por linha)
        cols = st.columns(3)
        for i, row in df_jogos.iterrows():
            with cols[i % 3]:
                # Usa st.container() que aplica o CSS do .game-card através de classes Streamlit
                with st.container():
                    # Imagem e informações do jogo
                    game_image_url = get_game_image(row["nome_jogo"])
                    st.image(game_image_url, use_container_width=True)
                    st.markdown(f"**{row['nome_jogo']}**")
                    st.caption(f"{row['caracteristica_1']} | {row['caracteristica_2']}")

                    # Slider de avaliação
                    try:
                        current_rating = df_matriz_utilidade.loc[
                            st.session_state.user_email, row["nome_jogo"]
                        ]
                        current_rating = (
                            int(current_rating) if pd.notnull(current_rating) else 0
                        )
                    except (KeyError, ValueError):
                        current_rating = 0

                    rating = st.slider(
                        "Avaliação",
                        min_value=0,
                        max_value=5,
                        value=current_rating,
                        key=f"rating_{i}",
                    )

                    # Exibe estrelas baseado na avaliação
                    if rating > 0:
                        st.markdown(
                            f'<div class="rating-display">{"★" * rating}{"☆" * (5-rating)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.user_ratings[row["nome_jogo"]] = rating
                        df_matriz_utilidade.loc[
                            st.session_state.user_email, row["nome_jogo"]
                        ] = rating
                    else:
                        st.write("Sem avaliação")
                        st.session_state.user_ratings.pop(row["nome_jogo"], None)
                        df_matriz_utilidade.loc[
                            st.session_state.user_email, row["nome_jogo"]
                        ] = 0

        if st.button("Salvar Avaliações"):
            # Salva as avaliações usando a função que atualiza o arquivo
            df_matriz_utilidade = save_ratings_to_file(
                st.session_state.user_email, st.session_state.user_ratings
            )
            # Limpa o cache para recarregar os dados salvos
            load_data.clear()
            st.success("Avaliações salvas!")
            # redireciona automaticamente para a página de Recomendações
            st.session_state.page = "recommendations"
            try:
                st.rerun()
            except Exception:
                st.info("Atualize a página (F5) para ver as recomendações.")
                st.stop()

        # botão visível que leva à página de recomendações sem salvar
        if st.button("Ver Recomendações"):
            st.session_state.page = "recommendations"
            try:
                st.rerun()
            except Exception:
                st.info("Atualize a página (F5) para ver as recomendações.")
                st.stop()

    except Exception as e:
        st.error(f"Erro ao carregar perfil: {str(e)}")
        st.session_state.page = "login"
        safe_rerun()

# Página de Recomendações
if st.session_state.page == "recommendations":
    st.header("Recomendações Personalizadas")

    # ----- Busca textual (TF-IDF) -----
    query = st.text_input(
        "O que você está procurando? (ex.: 'mundo aberto fantasia')", value=""
    )
    buscar = st.button("Buscar por Texto")

    # constrói vetorizador (cacheado) e matriz TF-IDF
    vec, X = build_tfidf_for_games(
        df_jogos,
        ngram_range=(1, 2),
        min_df=1,
        weight_map={"caracteristica_1": 2, "caracteristica_3": 2},
    )

    if buscar and query.strip():
        results_text = recommend_by_text(query, df_jogos, vec, X, top_n=10)
        if not results_text:
            st.info("Nenhum resultado encontrado para a consulta.")
        else:
            st.subheader("Resultados da busca por texto")
            for nome, score in results_text:
                st.markdown(f"**{nome}** — similaridade: {score:.3f}")

    # ----- Recomendações por perfil (se usuário logado) -----
    if st.session_state.user_email:
        st.write("---")
        st.subheader("Recomendações baseadas no seu perfil")
        perfil_rec = get_recommendations(
            st.session_state.user_email, df_jogos, df_matriz_utilidade, top_n=10
        )
        if not perfil_rec:
            st.info(
                "Avalie pelo menos 3 jogos e salve para obter recomendações de perfil."
            )
        else:
            for nome, score in perfil_rec:
                st.markdown(f"**{nome}** — relevância: {score:.3f}")

    # ----- Botões de navegação -----
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Voltar para Avaliações"):
            st.session_state.page = "rating"
            try:
                st.rerun()
            except Exception:
                st.info("Atualize a página (F5).")
                st.stop()
# Página de Busca e Recomposição
if st.session_state.page == "search_recombine":
    st.header("Busca por Texto e Recomposição")
    query = st.text_input("O que você procura?", value="")
    col1, col2 = st.columns([3, 1])
    with col2:
        alpha = st.slider("Peso do texto (alpha)", 0.0, 1.0, 0.6, 0.05)

    # gera scores por texto e por perfil
    vec, X = build_tfidf_for_games(df_jogos)  # já existente
    scores_text = {}
    if query.strip():
        text_results = recommend_by_text(query, df_jogos, vec, X, top_n=len(df_jogos))
        scores_text = {n: s for n, s in text_results}

    scores_profile = {}
    profile_results = get_recommendations(
        st.session_state.user_email, df_jogos, df_matriz_utilidade, top_n=len(df_jogos)
    )
    scores_profile = {n: s for n, s in profile_results}

    # combinação simples
    combined = {}
    for name in df_jogos["nome_jogo"]:
        t = scores_text.get(name, 0.0)
        p = scores_profile.get(name, 0.0)
        combined[name] = alpha * t + (1 - alpha) * p

    top_n = st.slider("Número de resultados combinados", 1, 20, 5)
    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]

    st.write("Resultados combinados:")
    for nome, score in top:
        st.markdown(f"**{nome}** — score combinado: {score:.4f}")
