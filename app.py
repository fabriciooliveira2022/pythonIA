import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2

from langchain_community.llms import Ollama

# --- CONFIG ---
st.set_page_config(page_title="Dashboard BI PRO", layout="wide")
st.title("📊 Dashboard Inteligente (Estilo Power BI)")

# --- LLM ---
@st.cache_resource
def load_llm():
    return Ollama(model="llama3:8b", temperature=0)

llm = load_llm()

# --- PROMPT PADRÃO PT-BR ---
def prompt_ptbr(texto):
    return f"""
    Você é um analista de dados especialista.

    ⚠️ Responda SEMPRE em português do Brasil.
    ⚠️ Nunca responda em inglês.

    Seja direto, claro e objetivo.

    {texto}
    """

# --- CONEXÃO ---
@st.cache_resource
def conectar():
    try:
        return psycopg2.connect(
            host="127.0.0.1",
            port="5432",
            database="listadecompras",
            user="postgres",
            password="123"
        )
    except Exception as e:
        st.error(f"Erro ao conectar no banco: {e}")
        return None

conn = conectar()

# --- TABELAS ---
@st.cache_data
def listar_tabelas():
    if conn is None:
        return []
    query = """
    SELECT table_name 
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
    """
    return pd.read_sql(query, conn)["table_name"].tolist()

# --- COLUNAS ---
@st.cache_data
def listar_colunas(tabela):
    if conn is None:
        return []
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %s
    """
    return pd.read_sql(query, conn, params=(tabela,))["column_name"].tolist()

# --- DADOS ---
@st.cache_data
def carregar_dados(tabela):
    if conn is None:
        return pd.DataFrame()
    query = f"SELECT * FROM {tabela} LIMIT 5000"
    return pd.read_sql(query, conn)

# --- LIMPEZA ---
def limpar_dados(df):
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

# --- IA ---
def perguntar_rapido(df, pergunta):
    contexto = df.head(30).to_string()

    prompt = prompt_ptbr(f"""
    Dados:
    {contexto}

    Pergunta: {pergunta}
    """)

    return llm.invoke(prompt)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configurações")

tabelas = listar_tabelas()

if not tabelas:
    st.warning("Nenhuma tabela encontrada.")
    st.stop()

tabela = st.sidebar.selectbox("Tabela", tabelas)

df = carregar_dados(tabela)

if df.empty:
    st.warning("Tabela sem dados.")
    st.stop()

df = limpar_dados(df)

# --- FILTROS ---
st.sidebar.subheader("🎛️ Filtros")

filtros = {}

for col in df.select_dtypes(include="object").columns:
    valores = st.sidebar.multiselect(
        col,
        df[col].dropna().unique()
    )
    if valores:
        filtros[col] = valores

for col, valores in filtros.items():
    df = df[df[col].isin(valores)]

# --- KPIs ---
st.subheader("📈 Indicadores")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros", len(df))
col2.metric("Colunas", df.shape[1])
col3.metric("Nulos", int(df.isnull().sum().sum()))

colunas_numericas = df.select_dtypes(include="number").columns

if len(colunas_numericas) > 0:
    col_num = colunas_numericas[0]
    col4.metric(f"Média {col_num}", round(df[col_num].mean(), 2))
else:
    col4.metric("Média", "N/A")

# --- GRÁFICOS ---
st.subheader("📊 Análises")

colA, colB = st.columns(2)

colunas_categoricas = df.select_dtypes(include="object").columns

# GRÁFICO 1
with colA:
    if len(colunas_categoricas) > 0:
        col_cat = st.selectbox("Coluna categórica", colunas_categoricas)

        contagem = df[col_cat].value_counts().reset_index()
        contagem.columns = [col_cat, "Quantidade"]

        fig1 = px.pie(contagem, names=col_cat, values="Quantidade", hole=0.4)
        fig1.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Sem colunas categóricas")

# GRÁFICO 2
with colB:
    if len(colunas_categoricas) > 0:
        col_cat2 = st.selectbox("Outra coluna", colunas_categoricas, key=2)

        contagem2 = df[col_cat2].value_counts().reset_index()
        contagem2.columns = [col_cat2, "Quantidade"]

        fig2 = px.bar(contagem2, x=col_cat2, y="Quantidade")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sem colunas categóricas")

# --- NUMÉRICO ---
st.subheader("📉 Distribuição Numérica")

if len(colunas_numericas) > 0:
    col_escolhida = st.selectbox("Coluna numérica", colunas_numericas)

    fig3 = px.histogram(df, x=col_escolhida)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Sem colunas numéricas")

# --- TABELA ---
st.subheader("📋 Dados")
st.dataframe(df.head(100))

# --- IA ---
st.subheader("🤖 Insights com IA")

if st.button("Gerar Insights"):
    df_sample = df.sample(min(len(df), 100))

    prompt = prompt_ptbr(f"""
    Gere insights claros, objetivos e acionáveis com base nos dados:

    {df_sample.head(20).to_string()}
    """)

    insight = llm.invoke(prompt)
    st.info(insight)

# --- CHAT ---
st.subheader("💬 Chat IA")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_user := st.chat_input("Pergunte algo sobre os dados..."):
    st.session_state.messages.append({"role": "user", "content": prompt_user})

    with st.chat_message("assistant"):
        resposta = perguntar_rapido(df, prompt_user)
        st.markdown(resposta)

        st.session_state.messages.append({
            "role": "assistant",
            "content": resposta
        })