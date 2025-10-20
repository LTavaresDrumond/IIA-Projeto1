# app.py
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== Config =====================
st.set_page_config(page_title="Sistema de Recomendação de Jogos", layout="wide")
st.markdown(
    """
<style>
    .stButton > button {
        background: none;
        border: none;
        padding: 0;
        font-size: 24px;
        line-height: 1;
        cursor: pointer;
    }
    .stButton > button:hover { color: gold; }
    .game-card {
        background-color: #ffffff; border-radius: 10px; padding: 20px; margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .rating-display { color: gold; font-size: 20px; margin-top: 5px; }
</style>
""",
    unsafe_allow_html=True,
)


# ===================== Estado & Navegação =====================
def init_state():
    st.session_state.setdefault("page", "login")
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("user_ratings", {})


def navigate(page: str):
    st.session_state.page = page
    try:
        st.rerun()
    except Exception:
        st.info("Atualize a página (F5)")
        st.stop()


def clean_email(email: str) -> str:
    return email.strip().lower() if email else ""


# ===================== Dados =====================
def _normalize_matrix_indices(df_matriz):
    df_matriz.index = df_matriz.index.to_series().astype(str).str.strip().str.lower()
    if df_matriz.index.duplicated().any():
        df_matriz = df_matriz.groupby(df_matriz.index).max()
    return df_matriz


def _align_matrix_columns(df_matriz, jogos):
    for j in jogos:
        if j not in df_matriz.columns:
            df_matriz[j] = 0.0
    return df_matriz.reindex(columns=jogos, fill_value=0.0)


@st.cache_data
def load_data():
    if not os.path.exists("dados_jogos.csv"):
        raise FileNotFoundError("dados_jogos.csv não encontrado.")
    df_jogos = pd.read_csv("dados_jogos.csv")
    jogos = df_jogos["nome_jogo"].tolist()

    if os.path.exists("matriz_utilidade.csv"):
        df_matriz = pd.read_csv("matriz_utilidade.csv", index_col=0)
        df_matriz = _normalize_matrix_indices(df_matriz)
        df_matriz = _align_matrix_columns(df_matriz, jogos)
    else:
        df_matriz = pd.DataFrame(columns=jogos)
        df_matriz.to_csv("matriz_utilidade.csv")

    return df_jogos, df_matriz


# ===================== Recomendação =====================
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
    return cosine_similarity(matriz_tfidf)


def _get_user_ratings(user_email, df_matriz):
    row = df_matriz.loc[user_email]
    return {col: int(v) for col, v in row.items() if pd.notnull(v) and float(v) > 0}


def _calculate_recommendation_scores(user_ratings, df_jogos, sim):
    name_to_idx = {name: idx for idx, name in enumerate(df_jogos["nome_jogo"])}
    scores = np.zeros(len(df_jogos))
    sim_sums = np.zeros(len(df_jogos))
    for nome, rating in user_ratings.items():
        if nome in name_to_idx:
            i = name_to_idx[nome]
            sim_col = sim[i]
            scores += sim_col * rating
            sim_sums += sim_col
    return scores, sim_sums, name_to_idx


def _get_final_predictions(scores, sim_sums, user_ratings, name_to_idx):
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.divide(scores, sim_sums)
        pred[np.isnan(pred)] = 0
    for nome in user_ratings:
        if nome in name_to_idx:
            pred[name_to_idx[nome]] = -1e9
    return pred


def get_recommendations(user_email, df_jogos, df_matriz, top_n=5):
    if not user_email or user_email not in df_matriz.index:
        return []
    user_ratings = _get_user_ratings(user_email, df_matriz)
    if not user_ratings:
        return []
    sim = calcular_similaridade_jogos(df_jogos)
    scores, sim_sums, name_to_idx = _calculate_recommendation_scores(
        user_ratings, df_jogos, sim
    )
    pred = _get_final_predictions(scores, sim_sums, user_ratings, name_to_idx)
    top_idx = np.argsort(-pred)[:top_n]
    return [(df_jogos.loc[i, "nome_jogo"], float(pred[i])) for i in top_idx]


# ===================== Páginas =====================
def page_login(df_jogos, df_matriz_utilidade):
    st.subheader("Login")
    with st.form("login_form"):
        email = st.text_input("E-mail")
        st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")

    if not submitted:
        st.info("Digite e-mail e senha para criar uma conta (protótipo)")
        return

    if not email.strip():
        st.error("Digite um e-mail válido")
        return

    clean_user_email = clean_email(email)
    st.session_state.user_email = clean_user_email

    # Garante o usuário na matriz
    if clean_user_email not in df_matriz_utilidade.index:
        nova_linha = pd.Series(0, index=df_jogos["nome_jogo"], name=clean_user_email)
        df_matriz_utilidade = pd.concat([df_matriz_utilidade, nova_linha.to_frame().T])
        df_matriz_utilidade.to_csv("matriz_utilidade.csv")
        st.success("Novo usuário criado!")

    navigate("rating")


def _rating_header():
    col1, col2 = st.columns([3, 1])
    with col2:
        st.write(f"Usuário: {st.session_state.user_email}")
        if st.button("Logout"):
            st.session_state.user_email = None
            st.session_state.user_ratings = {}
            navigate("login")


def _get_current_rating(df_matriz_utilidade, user_email, game_name):
    try:
        current_rating = df_matriz_utilidade.loc[user_email, game_name]
        return int(current_rating) if pd.notnull(current_rating) else 0
    except (KeyError, ValueError):
        return 0


def _update_game_rating(rating, game_name, df_matriz_utilidade, user_email):
    if rating > 0:
        st.markdown(
            f'<div class="rating-display">{"★" * rating}{"☆" * (5-rating)}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.user_ratings[game_name] = rating
        df_matriz_utilidade.loc[user_email, game_name] = rating
    else:
        st.write("Sem avaliação")
        st.session_state.user_ratings.pop(game_name, None)
        df_matriz_utilidade.loc[user_email, game_name] = 0


def _render_game_card(i, row, df_matriz_utilidade, user_email):
    st.markdown('<div class="game-card">', unsafe_allow_html=True)
    st.markdown(f"**{row['nome_jogo']}**")
    st.caption(f"{row['caracteristica_1']} | {row['caracteristica_2']}")

    current_rating = _get_current_rating(
        df_matriz_utilidade, user_email, row["nome_jogo"]
    )
    rating = st.slider(
        "Avaliação", min_value=0, max_value=5, value=current_rating, key=f"rating_{i}"
    )
    _update_game_rating(rating, row["nome_jogo"], df_matriz_utilidade, user_email)

    st.markdown("</div>", unsafe_allow_html=True)


def page_rating(df_jogos, df_matriz_utilidade):
    if st.session_state.user_email is None:
        st.warning("Faça login primeiro.")
        if st.button("Ir para Login"):
            navigate("login")
        return

    _rating_header()
    st.subheader("Avalie os Jogos")

    cols = st.columns(3)
    user_email = st.session_state.user_email
    for i, row in df_jogos.iterrows():
        with cols[i % 3]:
            with st.container():
                _render_game_card(i, row, df_matriz_utilidade, user_email)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Salvar Avaliações"):
            df_matriz_utilidade.to_csv("matriz_utilidade.csv")
            st.success("Avaliações salvas!")
            navigate("recommendations")
    with c2:
        if st.button("Ver Recomendações (sem salvar)"):
            navigate("recommendations")


def _load_utility_matrix(df_jogos):
    df_matriz_utilidade = pd.read_csv("matriz_utilidade.csv", index_col=0)
    df_matriz_utilidade.index = (
        df_matriz_utilidade.index.to_series().astype(str).str.strip().str.lower()
    )
    df_matriz_utilidade = df_matriz_utilidade.reindex(
        columns=df_jogos["nome_jogo"].tolist(), fill_value=0.0
    )
    return df_matriz_utilidade


def _check_user_authenticated():
    user = (st.session_state.user_email or "").strip().lower()
    if not user:
        st.warning("Faça login e avalie alguns jogos antes de ver recomendações.")
        if st.button("Ir para Login"):
            navigate("login")
        return None
    return user


def _display_recommendations(recs):
    if not recs:
        st.info(
            "Não há recomendações — avalie pelo menos 3 jogos e salve as avaliações."
        )
    else:
        for nome, score in recs:
            st.markdown(f"**{nome}** — relevância: {score:.3f}")


def page_recommendations(df_jogos):
    st.header("Recomendações Personalizadas")

    user = _check_user_authenticated()
    if user is None:
        return

    try:
        df_matriz_utilidade = _load_utility_matrix(df_jogos)
    except Exception:
        st.error("Erro ao carregar avaliações salvas.")
        return

    topn = st.slider("Número de recomendações", 1, 10, 5)
    recs = get_recommendations(user, df_jogos, df_matriz_utilidade, top_n=topn)

    _display_recommendations(recs)

    if st.button("Voltar para Avaliações"):
        navigate("rating")


# ===================== App =====================
def main():
    init_state()
    st.title("Sistema de Recomendação de Jogos")

    # Carrega dados (cacheado)
    df_jogos, df_matriz_utilidade = load_data()

    # Roteamento sem condicionais gigantes
    PAGES = {
        "login": lambda: page_login(df_jogos, df_matriz_utilidade),
        "rating": lambda: page_rating(df_jogos, df_matriz_utilidade),
        "recommendations": lambda: page_recommendations(df_jogos),
    }
    PAGES.get(st.session_state.page, PAGES["login"])()


if __name__ == "__main__":
    main()
