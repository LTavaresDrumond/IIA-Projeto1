# app.py

import time
import itertools
import io
import datetime
from pathlib import Path
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
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

# define caminho base do projeto (relativo ao arquivo app.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  # use BASE_DIR / "data" se preferir subpasta

# pasta para salvar resultados (usa DATA_DIR se definido)
RESULTS_DIR = Path(DATA_DIR) if "DATA_DIR" in globals() else Path(__file__).resolve().parent
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRID_CSV = RESULTS_DIR / "evaluation_grid_results.csv"
SINGLE_CSV = RESULTS_DIR / "evaluation_last_run.csv"


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


def save_metrics_row(params: dict, metrics: dict, path: Path):
    row = {**params, **metrics}
    row["timestamp"] = datetime.datetime.utcnow().isoformat()
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def run_grid_search(
    df_jogos,
    df_matriz,
    ngram_options,
    min_df_options,
    weight_maps,
    min_ratings,
    n_test,
    top_k,
    sample_users,
):
    results = []
    total = len(ngram_options) * len(min_df_options) * len(weight_maps)
    i = 0
    for ngram in ngram_options:
        for min_df in min_df_options:
            for wmap in weight_maps:
                i += 1
                params = {
                    "ngram_range": f"{ngram[0]}-{ngram[1]}",
                    "min_df": int(min_df),
                    "weight_map": str(wmap),
                }
                metrics = evaluate_offline_streamlit(
                    df_jogos,
                    df_matriz,
                    ngram_range=ngram,
                    min_df=int(min_df),
                    weight_map=wmap,
                    min_ratings=min_ratings,
                    n_test=n_test,
                    top_k=top_k,
                    sample_users=sample_users,
                )
                results.append({**params, **metrics})
    df_res = pd.DataFrame(results)
    return df_res


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

    /* Game card styling */
    .game-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
                with st.container():
                    st.markdown('<div class="game-card">', unsafe_allow_html=True)

                    # Imagem e informações do jogo
                    # st.image("https://placehold.in/200@2x", use_container_width=True)
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

                    st.markdown("</div>", unsafe_allow_html=True)

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

    st.write("---")
    st.subheader("Grid‑Search e Salvamento de Resultados")

    with st.expander("Grid‑Search TF‑IDF (rodar várias configurações)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            # ngramas (opções rápidas)
            ngram_choice = st.multiselect(
                "n‑gram options (selecione pares)", options=["1,1", "1,2"], default=["1,2"]
            )
            ngram_options = [
                (int(s.split(",")[0]), int(s.split(",")[1])) for s in ngram_choice
            ]
            min_df_list = st.multiselect(
                "min_df options", options=[1, 2, 3, 5], default=[1, 2]
            )
        with col2:
            # weight_map presets (usuário pode editar)
            preset = st.selectbox(
                "Preset weight_map", options=["gênero+tag (2,2)", "gênero(2)", "sem peso extra"], index=0
            )
            if preset == "gênero+tag (2,2)":
                weight_maps = [
                    {"caracteristica_1": 2, "caracteristica_3": 2},
                    {"caracteristica_1": 3, "caracteristica_3": 1},
                ]
            elif preset == "gênero(2)":
                weight_maps = [{"caracteristica_1": 2}, {"caracteristica_1": 3}]
            else:
                weight_maps = [{}]
            min_ratings_g = st.number_input(
                "min_ratings (usuários elegíveis)", min_value=1, max_value=50, value=5
            )
            n_test_g = st.number_input(
                "n_test por usuário", min_value=1, max_value=5, value=1
            )
            top_k_g = st.number_input(
                "top_k (metrics K)", min_value=1, max_value=50, value=10
            )
            sample_users_g = st.number_input(
                "sample users (0 = todos)", min_value=0, max_value=1000, value=0
            )
        run_grid = st.button("Executar Grid‑Search")

        if run_grid:
            st.info("Executando grid‑search — isso pode demorar.")
            with st.spinner("Executando combinações..."):
                df_grid = run_grid_search(
                    df_jogos,
                    df_matriz_utilidade,
                    ngram_options,
                    min_df_list,
                    weight_maps,
                    min_ratings=int(min_ratings_g),
                    n_test=int(n_test_g),
                    top_k=int(top_k_g),
                    sample_users=int(sample_users_g),
                )
            st.success("Grid‑search concluído")
            st.dataframe(
                df_grid.sort_values(by=["precision"], ascending=False).reset_index(drop=True)
            )

            # salvar CSV e oferecer download
            df_grid.to_csv(GRID_CSV, index=False)
            csv_bytes = df_grid.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar CSV (grid results)",
                data=csv_bytes,
                file_name=GRID_CSV.name,
                mime="text/csv",
            )

            # plot: precision/recall/map por configuração
            df_long = df_grid.melt(
                id_vars=["ngram_range", "min_df", "weight_map"],
                value_vars=["precision", "recall", "map"],
                var_name="metric",
                value_name="value",
            )
            chart = alt.Chart(df_long).mark_line(point=True).encode(
                x=alt.X("ngram_range:N", title="ngram_range"),
                y=alt.Y("value:Q", title="valor"),
                color="metric:N",
                strokeDash="metric:N",
                column=alt.Column(
                    "min_df:N", header=alt.Header(title="min_df")
                ),
            ).properties(height=200, width=200)
            st.altair_chart(chart, use_container_width=True)

            # salvar plot como PNG (render via savechart)
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 3))
                for m in ["precision", "recall", "map"]:
                    grp = df_grid.groupby("ngram_range")[m].mean()
                    ax.plot([str(x) for x in grp.index], grp.values, marker="o", label=m)
                ax.set_xlabel("ngram_range")
                ax.set_ylabel("valor médio")
                ax.legend()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    "Baixar gráfico (PNG)",
                    data=buf,
                    file_name="grid_metrics.png",
                    mime="image/png",
                )
            except Exception:
                # matplotlib pode não estar disponível, mas Altair já mostra gráfico
                pass

    # --- salvar resultado de execução única da avaliação ---
    with st.expander("Salvar resultado da última avaliação", expanded=False):
        st.write(
            "Se desejar, salve o último resultado exibido (ou execute avaliação e salve)."
        )
        if st.button("Salvar última execução"):
            # tenta recuperar parâmetros mostrados antes (se existirem)
            # como fallback, roda uma avaliação rápida com defaults
            metrics = evaluate_offline_streamlit(
                df_jogos,
                df_matriz_utilidade,
                ngram_range=(1, 2),
                min_df=1,
                weight_map={"caracteristica_1": 2, "caracteristica_3": 2},
                min_ratings=5,
                n_test=1,
                top_k=10,
                sample_users=50,
            )
            params = {
                "ngram_range": "1-2",
                "min_df": 1,
                "weight_map": "{'caracteristica_1':2,'caracteristica_3':2}",
            }
            save_metrics_row(params, metrics, SINGLE_CSV)
            st.success(f"Resultado salvo em {SINGLE_CSV.name}")
            with open(SINGLE_CSV, "rb") as f:
                st.download_button(
                    "Baixar CSV (última execução)",
                    data=f,
                    file_name=SINGLE_CSV.name,
                    mime="text/csv",
                )

    st.write("---")
    st.subheader("Trecho para o relatório (metodologia TF‑IDF)")
    st.markdown(
        """
- Pré-processamento: concatenação das características (caracteristica_1..5), remoção de stopwords (pt+en), n-grams (1,2).
- Vetorizador: TfidfVectorizer (sublinear_tf=True), parâmetros testados: ngram_range ∈ {(1,1),(1,2)}, min_df ∈ {1,2,3,5}, weight_map aplicado repetindo campos.
- Estratégia de avaliação offline: leave-one-out (n_test itens por usuário) sobre usuários com ≥ min_ratings; métricas calculadas: Precision@K, Recall@K, MAP@K.
- Combinação híbrida (se aplicável): score_final = α·score_text + (1−α)·score_profile; α ajustado empiricamente.
"""
    )

# ------- Debug / inspeção do Grid (apêndice na página de Recomendações) -------
import ast

def parse_ngram_string(s: str):
    # aceita formatos "1,2" ou "1-2" ou tupla já serializada
    if isinstance(s, (list, tuple)):
        return tuple(s)
    if isinstance(s, str):
        if ',' in s:
            a,b = s.split(',')
            return (int(a.strip()), int(b.strip()))
        if '-' in s:
            a,b = s.split('-')
            return (int(a.strip()), int(b.strip()))
    return (1,2)

def parse_weight_map(s: str):
    # tenta converter string para dict (se já for dict retorna)
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        # fallback: vazio
        return {}

def inspect_grid_row_and_show_games(df_row, df_jogos, df_matriz, top_n=5, sample_user=None):
    """
    Dado um dicionário/row com parâmetros (ngram_range, min_df, weight_map),
    reconstrói TF-IDF e mostra top-N recomendações para sample_user.
    """
    ngram = parse_ngram_string(df_row.get('ngram_range', '1-2'))
    min_df = int(df_row.get('min_df', 1))
    wmap = parse_weight_map(df_row.get('weight_map', '{}'))
    # constroi tfidf com esses parametros
    vec, X = build_tfidf_for_games(df_jogos, ngram_range=ngram, min_df=min_df, weight_map=wmap)
    # escolhe usuário exemplo
    users = [u for u in df_matriz.index if (df_matriz.loc[u] > 0).sum() > 0]
    if not users:
        st.info("Nenhum usuário com avaliações para inspecionar.")
        return
    user = sample_user if sample_user in users else users[0]
    recs = recommend_for_user_from_train(user, df_jogos, df_matriz, X, top_n=top_n)
    st.markdown(f"**Configuração selecionada:** ngram={ngram}, min_df={min_df}, weight_map={wmap}")
    st.markdown(f"**Usuário exemplo:** {user} — top {top_n} recomendações")
    if not recs:
        st.write("Nenhuma recomendação gerada para este usuário/configuração.")
    else:
        for r in recs:
            st.write(f"- {r}")

# Explanatory text for the UI variables
st.markdown("### Explicação dos parâmetros (para o relatório)")
st.markdown("""
- min_ratings: número mínimo de avaliações que um usuário precisa ter para ser incluído na avaliação offline. Define quem é elegível para o holdout.
- n_test: quantos itens por usuário são removidos (holdout) para testar. Com n_test=1 usamos leave‑one‑out.
- sample_users: se >0, uma amostra aleatória de usuários será usada para acelerar a avaliação.
- top_k: k usado para calcular Precision@K/Recall@K/MAP@K (quanto maior, mais tolerante).
- ngram_range: (1,1) = apenas unigrams; (1,2) = unigrams + bigrams (capta expressões como "mundo aberto").
- min_df: termo mínimo de documentos para considerar uma feature no TF‑IDF (ajuda a remover termos raros).
- weight_map: dicionário que repete campos na construção do documento (ex.: {'caracteristica_1':2} aumenta o peso do gênero). Use valores inteiros ≥1.
""")

st.markdown("### Por que o Grid‑Search retorna métricas e não títulos de jogos?")
st.markdown("""
O Grid‑Search testa combinações de parâmetros e devolve métricas agregadas (Precision/Recall/MAP) — essas métricas resumem a qualidade do ranking para cada configuração, não os títulos individuais.  
Se quiser ver os títulos recomendados para uma configuração específica, use a ferramenta de inspeção abaixo: selecione a linha do resultado e escolha um usuário exemplo; o app mostrará os jogos recomendados (top‑N) para essa configuração.
""")

# If grid results file exists, allow inspection
if (RESULTS_DIR / "evaluation_grid_results.csv").exists():
    try:
        df_grid = pd.read_csv(RESULTS_DIR / "evaluation_grid_results.csv")
        st.write("---")
        st.subheader("Inspecionar resultados do Grid (mostrar títulos recomendados por configuração)")
        st.write("Escolha uma linha do CSV de resultados (ou selecione manualmente os parâmetros) e veja top‑N jogos recomendados para um usuário exemplo.")
        if not df_grid.empty:
            st.dataframe(df_grid.reset_index(drop=True))
            idx = st.number_input("Índice da linha do grid para inspeção", min_value=0, max_value=max(0, len(df_grid)-1), value=0)
            top_n_inspect = st.number_input("Top N jogos a mostrar", min_value=1, max_value=20, value=5)
            # user selector (lista curta)
            users = [u for u in df_matriz_utilidade.index if (df_matriz_utilidade.loc[u] > 0).sum() > 0]
            sample_user = st.selectbox("Usuário exemplo (para ver recomendações)", options=users[:200] if users else ["nenhum"])
            row = df_grid.iloc[int(idx)].to_dict()
            if st.button("Mostrar recomendações (para a configuração selecionada)"):
                inspect_grid_row_and_show_games(row, df_jogos, df_matriz_utilidade, top_n=int(top_n_inspect), sample_user=sample_user)
    except Exception as e:
        st.write("Não foi possível carregar o CSV de grid para inspeção:", str(e))

# breve nota final
st.markdown("___")
st.markdown("Se preferir, posso também: (a) fazer com que o grid salve, para cada configuração, os *top games* médios; (b) exportar um relatório completo com as melhores configurações e exemplos de recomendações. Diga qual prefere.")
