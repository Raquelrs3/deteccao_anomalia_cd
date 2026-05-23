import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Gastos Parlamentares",
    page_icon="\U0001f3db",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f0f4f0; }
.main { background-color: #f0f4f0; }

/* impede colapso da sidebar */
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 240px !important;
    transform: none !important;
}

section[data-testid="stSidebar"] {
    background: #2c2f33;
    border-right: 1px solid #3a3d42;
}
section[data-testid="stSidebar"] * { color: #b0b8c1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e8ecf0 !important; }

/* selectbox na sidebar */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background-color: #3a3d42 !important;
    border-color: #4a4d52 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
    color: #e8ecf0 !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #7a8290 !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stSelectbox [role="listbox"] {
    background-color: #3a3d42 !important;
}
section[data-testid="stSidebar"] .stSelectbox [role="option"] {
    background-color: #3a3d42 !important;
    color: #e8ecf0 !important;
}
section[data-testid="stSidebar"] .stSelectbox [role="option"]:hover {
    background-color: #107ae2 !important;
}

/* nav radio */
section[data-testid="stSidebar"] .stRadio label {
    color: #b0b8c1 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 4px 0 !important;
}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked ~ div {
    border-color: #107ae2 !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

.kpi-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 14px 18px;
    border: 1px solid #dde8dd;
    margin-bottom: 0;
    border-left: 3px solid #107ae2;
}
.kpi-card.red    { border-left-color: #c0392b; }
.kpi-card.green  { border-left-color: #828a82; }
.kpi-card.orange { border-left-color: #e3ece3; border-left-width: 3px; }

.kpi-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #828a82;
    margin-bottom: 5px;
    line-height: 1;
}
.kpi-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.kpi-sub { font-size: 0.65rem; color: #828a82; margin-top: 4px; }
.kpi-alert { font-size: 0.65rem; color: #c0392b; font-weight: 600; margin-top: 4px; }

.sec-title {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #828a82;
    margin: 18px 0 10px 0;
    padding-bottom: 7px;
    border-bottom: 1px solid #dde8dd;
}
.page-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.01em;
    margin-bottom: 2px;
}
.page-sub { font-size: 0.7rem; color: #828a82; margin-bottom: 18px; }
</style>
""", unsafe_allow_html=True)

# ── PALETA
C_BLUE   = "#107ae2"
C_GRAY   = "#828a82"
C_GREEN  = "#e3ece3"
C_DARK   = "#1a1a1a"
C_RED    = "#c0392b"
C_WARN   = "#d97706"

def plot_base(height=300, title=""):
    return dict(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", size=10, color="#444"),
        margin=dict(l=12, r=12, t=36 if title else 16, b=12),
        title_text=title,
        title_font=dict(size=11, color="#444", family="Inter"),
        title_x=0,
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    )

def xy_clean(fig, grid_x=False, grid_y=True, ticksize_x=10, ticksize_y=10):
    fig.update_xaxes(showgrid=grid_x, zeroline=False, linecolor="#dde8dd",
                     tickfont_size=ticksize_x)
    fig.update_yaxes(showgrid=grid_y, gridcolor="#f0f4f0", zeroline=False,
                     tickfont_size=ticksize_y)
    return fig

# ── CACHE
@st.cache_data
def carregar():
    dep  = pd.read_csv("deputados.csv", low_memory=False)
    desp = pd.read_csv("despesas.csv",  low_memory=False)
    for c in ["cnpj_cpf_fornecedor","url_documento","num_documento","nome_fornecedor"]:
        desp[c] = desp[c].fillna("NAO INFORMADO").astype(str)
    dep["partido"] = dep["partido"].fillna("SEM PARTIDO")
    dep["uf"]      = dep["uf"].fillna("NAO INFORMADO")
    dep            = dep.drop_duplicates(subset=["id_deputado"])
    desp["id_deputado"]    = desp["id_deputado"].astype(str)
    dep["id_deputado"]     = dep["id_deputado"].astype(str)
    desp["data_documento"] = pd.to_datetime(desp["data_documento"], errors="coerce")
    df = desp.merge(dep[["id_deputado","nome","partido","uf"]], on="id_deputado", how="left")
    df["partido"] = df["partido"].fillna("SEM PARTIDO")
    df["uf"]      = df["uf"].fillna("NAO INFORMADO")
    df["year"]    = df["data_documento"].dt.year
    df["month"]   = df["data_documento"].dt.month
    return df

@st.cache_data
def rodar_modelo(df_pos):
    stats = df_pos.groupby("tipo_despesa")["valor_liquido"].agg(
        mediana_categoria="median",
        media_categoria="mean",
        desvio_categoria="std"
    ).reset_index()
    df = df_pos.merge(stats, on="tipo_despesa", how="left")
    df["razao_mediana"]     = df["valor_liquido"] / (df["mediana_categoria"] + 1)
    df["z_score_categoria"] = df.groupby("tipo_despesa")["valor_liquido"].transform(
        lambda x: zscore(x, ddof=1)
    )
    le1, le2, le3 = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["te"] = le1.fit_transform(df["tipo_despesa"])
    df["pe"] = le2.fit_transform(df["partido"])
    df["ue"] = le3.fit_transform(df["uf"])
    feats = ["valor_liquido","valor_glosa","mediana_categoria","razao_mediana",
             "z_score_categoria","te","pe","ue","year","month"]
    X = RobustScaler().fit_transform(df[feats])
    m = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    df["anomalia"]       = m.fit_predict(X)
    df["anomaly_score"]  = m.decision_function(X)
    df["anomalia_label"] = df["anomalia"].map({1:"Normal", -1:"Anomalia"})
    return df

with st.spinner("Carregando e processando dados..."):
    df_raw = carregar()
    df_est = df_raw[df_raw["valor_liquido"] < 0].copy()
    df_pos = df_raw[df_raw["valor_liquido"] >= 0].copy().reset_index(drop=True)
    df     = rodar_modelo(df_pos)

# ── SIDEBAR
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.14em;
    text-transform:uppercase;color:#7a8290;margin-bottom:14px">Navegação</div>""",
    unsafe_allow_html=True)

    pagina = st.radio("", [
        "Visão Geral", "Por Categoria",
        "Anomalias", "Fornecedores", "Temporal", "Hipóteses"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#3a3d42;margin:18px 0'>", unsafe_allow_html=True)

    anos_disp     = sorted(df["ano"].dropna().unique().astype(int))
    partidos_disp = sorted(df["partido"].unique())
    ufs_disp      = sorted(df["uf"].unique())

    ano_sel = st.multiselect(
    "Ano",
    options=anos_disp,
    placeholder="Todos os anos"
    )

    partido_sel = st.multiselect(
        "Partido",
        options=partidos_disp,
        placeholder="Todos os partidos"
    )

    uf_sel = st.multiselect(
        "Estado (UF)",
        options=ufs_disp,
        placeholder="Todos os estados"
    )

    st.markdown("<hr style='border-color:#3a3d42;margin:18px 0'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.62rem;color:#7a8290;line-height:2">
    API Dados Abertos<br>Câmara dos Deputados<br>
    57ª Legislatura · 2023–2026<br>
    Modelo: Isolation Forest · 5%
    </div>""", unsafe_allow_html=True)

# ── FILTRO
anos_sel = ano_sel if ano_sel else anos_disp
partidos_sel = partido_sel if partido_sel else partidos_disp
ufs_sel = uf_sel if uf_sel else ufs_disp

mask   = df["ano"].isin(anos_sel) & df["partido"].isin(partidos_sel) & df["uf"].isin(ufs_sel)
df_f   = df[mask].copy()
anom_f = df_f[df_f["anomalia"] == -1].copy()
norm_f = df_f[df_f["anomalia"] ==  1].copy()

def kpi(label, value, sub="", cls=""):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return f"""<div class="kpi-card {cls}">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {sub_html}
    </div>"""

def kpi_alert(label, value, alert):
    return f"""<div class="kpi-card red">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-alert">{alert}</div>
    </div>"""

def sec(t):
    st.markdown(f"<div class='sec-title'>{t}</div>", unsafe_allow_html=True)

# ═══════════════════════════════
# VISÃO GERAL
# ═══════════════════════════════
if pagina == "Visão Geral":
    st.markdown("<div class='page-title'>Pipeline de Inteligência em Gastos Públicos</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>57ª Legislatura · API Dados Abertos da Câmara dos Deputados </div>", unsafe_allow_html=True)

    total  = len(df_f)
    val    = df_f["valor_liquido"].sum()
    n_anom = len(anom_f)
    taxa   = n_anom / total * 100 if total > 0 else 0
    v_anom = anom_f["valor_liquido"].sum()
    pct_v  = v_anom / val * 100 if val > 0 else 0
    med_a  = anom_f["valor_liquido"].mean() if len(anom_f) > 0 else 0
    med_n  = norm_f["valor_liquido"].mean()  if len(norm_f) > 0 else 1
    fator  = med_a / med_n if med_n > 0 else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi("Registros", f"{total:,.0f}", f"{df_f['id_deputado'].nunique()} deputados"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Volume Total", f"R$ {val/1e6:.1f}M", f"{df_f['nome_fornecedor'].nunique():,.0f} fornecedores"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_alert("Anomalias", f"{n_anom:,.0f}", f"↑ {taxa:.2f}% dos registros"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_alert("Valor em Anomalias", f"R$ {v_anom/1e6:.1f}M", f"↑ {pct_v:.1f}% do total"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_alert("Média — Anomalia", f"R$ {med_a:,.0f}", f"vs R$ {med_n:,.0f} (normal)"), unsafe_allow_html=True)
    with c6: st.markdown(kpi("Fator Discrepância", f"{fator:.1f}×", "anomalia vs normal", cls="green"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_p, col_c = st.columns([1,1,2])

    with col_g:
        sec("Taxa de Anomalia")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=taxa,
            number={"suffix":"%","font":{"size":26,"family":"Inter","color":C_DARK}},
            gauge={
                "axis":{"range":[0,15],"tickwidth":1,"tickcolor":"#dde8dd","tickfont":{"size":8}},
                "bar":{"color":C_RED,"thickness":0.2},
                "bgcolor":"white","borderwidth":0,
                "steps":[
                    {"range":[0,3],"color":C_GREEN},
                    {"range":[3,7],"color":"#fef9c3"},
                    {"range":[7,15],"color":"#fde8e8"},
                ],
            }
        ))
        fig_g.update_layout(**plot_base(220))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_p:
        sec("Classificação")
        pz = pd.DataFrame({"Tipo":["Normal","Anomalia"],"Qtd":[len(norm_f),len(anom_f)]})
        fig_p = px.pie(pz, names="Tipo", values="Qtd", hole=0.6,
                       color="Tipo",
                       color_discrete_map={"Normal":C_BLUE,"Anomalia":C_RED})
        fig_p.update_traces(textposition="outside", textinfo="percent+label",
                            textfont_size=9, marker_line_width=0)
        fig_p.update_layout(**plot_base(220), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    with col_c:
        sec("Normal vs Anomalia — Média · Mediana · Máximo (R$)")
        comp = pd.DataFrame({
            "Grupo":  ["Normal","Normal","Normal","Anomalia","Anomalia","Anomalia"],
            "Métrica":["Média","Mediana","Máximo","Média","Mediana","Máximo"],
            "Valor":  [norm_f["valor_liquido"].mean(), norm_f["valor_liquido"].median(), norm_f["valor_liquido"].max(),
                       anom_f["valor_liquido"].mean(), anom_f["valor_liquido"].median(), anom_f["valor_liquido"].max()],
        })
        fig_c = px.bar(comp, x="Métrica", y="Valor", color="Grupo", barmode="group",
                       color_discrete_map={"Normal":C_BLUE,"Anomalia":C_RED})
        fig_c.update_traces(marker_line_width=0)
        fig_c.update_layout(**plot_base(220, ""))
        fig_c = xy_clean(fig_c)
        fig_c.update_layout(legend_title_text="", xaxis_title="", yaxis_title="R$")
        st.plotly_chart(fig_c, use_container_width=True)

    sec("Distribuição do Anomaly Score — Normal × Anomalia")
    sn = norm_f["anomaly_score"].sample(min(30000, len(norm_f)), random_state=42)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Histogram(x=sn, name="Normal", marker_color=C_BLUE, opacity=0.55, nbinsx=100))
    fig_sc.add_trace(go.Histogram(x=anom_f["anomaly_score"], name="Anomalia", marker_color=C_RED, opacity=0.8, nbinsx=100))
    fig_sc.add_vline(x=0, line_dash="dot", line_color=C_GRAY, line_width=1,
                     annotation_text="limiar 0", annotation_font_size=8)
    fig_sc.update_traces(marker_line_width=0)
    fig_sc.update_layout(**plot_base(220, "Score negativo = maior anomalia"),
                         barmode="overlay")
    fig_sc = xy_clean(fig_sc)
    fig_sc.update_layout(xaxis_title="Anomaly Score", yaxis_title="Frequência",
                         legend=dict(orientation="h",yanchor="bottom",y=1.01,font=dict(size=9)))
    st.plotly_chart(fig_sc, use_container_width=True)
    
    sec("Principais Insights")

    cat_top = (
        df_f.groupby("tipo_despesa")["valor_liquido"]
            .sum()
            .sort_values(ascending=False)
    )

    top3_pct = (cat_top.head(3).sum() / cat_top.sum() * 100)

    top_forn = (
        df_f.groupby("nome_fornecedor")["valor_liquido"]
            .sum()
            .sort_values(ascending=False)
    )

    top20_pct = (top_forn.head(20).sum() / top_forn.sum() * 100)

    st.info(f"""
✓ As 3 maiores categorias concentram {top3_pct:.1f}% do total gasto

✓ Os 20 maiores fornecedores concentram {top20_pct:.1f}% dos recursos

✓ Foram detectadas {len(anom_f):,.0f} despesas anômalas

✓ O valor médio das anomalias é {fator:.1f}x maior que o das despesas normais

✓ {anom_f['id_deputado'].nunique()} deputados possuem registros classificados como anômalos
""")


# ═══════════════════════════════
# POR CATEGORIA
# ═══════════════════════════════
elif pagina == "Por Categoria":
    st.markdown("<div class='page-title'>Análise por Categoria de Despesa</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Volume gasto e presença de anomalias por tipo de despesa</div>", unsafe_allow_html=True)

    sec("Volume Total por Categoria (R$)")
    cat = (df_f.groupby("tipo_despesa")["valor_liquido"].sum()
               .reset_index().sort_values("valor_liquido"))
    cat["perc"] = (cat["valor_liquido"]/cat["valor_liquido"].sum()*100).round(1)
    fig_cl = px.bar(cat, x="valor_liquido", y="tipo_despesa",
                    orientation="h", text="perc",
                    color="valor_liquido",
                    color_continuous_scale=[[0,"#dde8dd"],[1,C_BLUE]])
    fig_cl.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                         textfont_size=8, marker_line_width=0)
    fig_cl.update_layout(**plot_base(480))
    fig_cl.update_coloraxes(showscale=False)
    fig_cl.update_xaxes(showgrid=False, zeroline=False, title="R$")
    fig_cl.update_yaxes(showgrid=False, zeroline=False, title="", tickfont_size=9)
    st.plotly_chart(fig_cl, use_container_width=True)

    sec("Anomalias por Categoria — Quantidade")
    cat_a = (anom_f.groupby("tipo_despesa")
                   .agg(qtd=("valor_liquido","count"), vm=("valor_liquido","mean"))
                   .reset_index().sort_values("qtd"))
    fig_cr = px.bar(cat_a, x="qtd", y="tipo_despesa",
                    orientation="h",
                    color="vm",
                    color_continuous_scale=[[0,"#fde8e8"],[1,C_RED]],
                    labels={"vm":"Valor Médio (R$)"})
    fig_cr.update_traces(marker_line_width=0)
    fig_cr.update_layout(**plot_base(480, "Cor = valor médio da anomalia"))
    fig_cr.update_coloraxes(colorbar=dict(title="Média R$",tickfont_size=8,len=0.5))
    fig_cr.update_xaxes(showgrid=False, zeroline=False, title="Quantidade")
    fig_cr.update_yaxes(showgrid=False, zeroline=False, title="", tickfont_size=9)
    st.plotly_chart(fig_cr, use_container_width=True)

# ═══════════════════════════════
# ANOMALIAS
# ═══════════════════════════════
elif pagina == "Anomalias":
    st.markdown("<div class='page-title'>Anomalias Detectadas</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Isolation Forest · contamination = 5% · 10 features</div>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(kpi_alert("Anomalias", f"{len(anom_f):,.0f}", f"↑ {len(anom_f)/max(len(df_f),1)*100:.2f}% do total"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_alert("Valor Total", f"R$ {anom_f['valor_liquido'].sum()/1e6:.1f}M", f"↑ {anom_f['valor_liquido'].sum()/max(df_f['valor_liquido'].sum(),1)*100:.1f}% da base"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_alert("Média por Anomalia", f"R$ {anom_f['valor_liquido'].mean():,.0f}", f"vs R$ {norm_f['valor_liquido'].mean():,.0f} (normal)"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_alert("Maior Anomalia", f"R$ {anom_f['valor_liquido'].max():,.0f}", "valor máximo detectado"), unsafe_allow_html=True)
    with c5: st.markdown(kpi("Deputados Envolvidos", f"{anom_f['id_deputado'].nunique()}", f"de {df_f['id_deputado'].nunique()} no filtro"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        sec("Anomalias por Partido (top 12)")
        pa = (anom_f.groupby("partido").agg(qtd=("valor_liquido","count"),val=("valor_liquido","sum"))
                    .sort_values("qtd",ascending=False).head(12).reset_index())
        fig_pa = px.bar(pa, x="qtd", y="partido", orientation="h",
                        color="val", color_continuous_scale=[[0,"#fde8e8"],[1,C_RED]],
                        labels={"val":"Valor Total (R$)"})
        fig_pa.update_traces(marker_line_width=0)
        fig_pa.update_layout(**plot_base(360))
        fig_pa.update_coloraxes(showscale=False)
        fig_pa.update_xaxes(showgrid=False, zeroline=False, title="Quantidade")
        fig_pa.update_yaxes(showgrid=False, zeroline=False, title="",
                            categoryorder="total ascending", tickfont_size=9)
        st.plotly_chart(fig_pa, use_container_width=True)

    with col_b:
        sec("Anomalias por Estado (top 15)")
        ua = (anom_f.groupby("uf").agg(qtd=("valor_liquido","count"),val=("valor_liquido","sum"))
                    .sort_values("qtd",ascending=False).head(15).reset_index())
        fig_ua = px.bar(ua, x="uf", y="qtd",
                        color="val", color_continuous_scale=[[0,"#fde8e8"],[1,C_RED]],
                        labels={"val":"Valor Total (R$)"})
        fig_ua.update_traces(marker_line_width=0)
        fig_ua.update_layout(**plot_base(360))
        fig_ua.update_coloraxes(showscale=False)
        fig_ua.update_xaxes(showgrid=False, zeroline=False, title="Estado")
        fig_ua.update_yaxes(showgrid=True, gridcolor="#f0f4f0", zeroline=False, title="Quantidade")
        st.plotly_chart(fig_ua, use_container_width=True)

    sec("Top 30 — Maiores Valores Anômalos")
    top = (anom_f[["tipo_despesa","partido","uf","nome_fornecedor",
                    "valor_liquido","mediana_categoria","razao_mediana","anomaly_score"]]
           .sort_values("valor_liquido",ascending=False).head(30).reset_index(drop=True))
    top.columns = ["Categoria","Partido","UF","Fornecedor","Valor (R$)","Mediana Cat","Razão","Score"]
    top["Valor (R$)"]   = top["Valor (R$)"].map(lambda x: f"R$ {x:,.2f}")
    top["Mediana Cat"]  = top["Mediana Cat"].map(lambda x: f"R$ {x:,.2f}")
    top["Razão"]        = top["Razão"].map(lambda x: f"{x:.1f}×")
    top["Score"]        = top["Score"].map(lambda x: f"{x:.3f}")
    st.dataframe(top, use_container_width=True, height=440)

# ═══════════════════════════════
# FORNECEDORES
# ═══════════════════════════════
elif pagina == "Fornecedores":
    st.markdown("<div class='page-title'>Análise de Fornecedores</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Concentração de recebimentos e presença em despesas anômalas</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sec("Top 20 Fornecedores — Volume Total")
        tf = (df_f.groupby("nome_fornecedor")["valor_liquido"].sum()
                  .reset_index().sort_values("valor_liquido").tail(20))
        tf["perc"] = (tf["valor_liquido"]/df_f["valor_liquido"].sum()*100).round(1)
        fig_tf = px.bar(tf, x="valor_liquido", y="nome_fornecedor",
                        orientation="h", text="perc",
                        color="valor_liquido",
                        color_continuous_scale=[[0,"#dde8dd"],[1,C_BLUE]])
        fig_tf.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                             textfont_size=8, marker_line_width=0)
        fig_tf.update_layout(**plot_base(480))
        fig_tf.update_coloraxes(showscale=False)
        fig_tf.update_xaxes(showgrid=False, zeroline=False, title="R$")
        fig_tf.update_yaxes(showgrid=False, zeroline=False, title="", tickfont_size=9)
        st.plotly_chart(fig_tf, use_container_width=True)

    with col_r:
        sec("Top 15 Fornecedores — Valor em Anomalias")
        fa = (anom_f.groupby("nome_fornecedor")["valor_liquido"].sum()
                    .reset_index().sort_values("valor_liquido").tail(15))
        fig_fa = px.bar(fa, x="valor_liquido", y="nome_fornecedor",
                        orientation="h",
                        color="valor_liquido",
                        color_continuous_scale=[[0,"#fde8e8"],[1,C_RED]])
        fig_fa.update_traces(marker_line_width=0)
        fig_fa.update_layout(**plot_base(480))
        fig_fa.update_coloraxes(showscale=False)
        fig_fa.update_xaxes(showgrid=False, zeroline=False, title="R$")
        fig_fa.update_yaxes(showgrid=False, zeroline=False, title="", tickfont_size=9)
        st.plotly_chart(fig_fa, use_container_width=True)

    sec("Concentração Acumulada de Fornecedores (Top 30)")
    fc = (df_f.groupby("nome_fornecedor")["valor_liquido"].sum()
              .reset_index().sort_values("valor_liquido", ascending=False))
    fc["pct_acum"] = fc["valor_liquido"].cumsum() / fc["valor_liquido"].sum() * 100
    fc["rank"]     = range(1, len(fc) + 1)
    fc30 = fc[fc["rank"] <= 30].copy()

    fig_fc = px.area(
        fc30, x="nome_fornecedor", y="pct_acum",
        hover_data={"valor_liquido": ":,.2f", "pct_acum": ":.1f"},
        labels={"nome_fornecedor": "Fornecedor",
                "pct_acum": "% Acumulado do Total Gasto",
                "valor_liquido": "Valor (R$)"}
    )
    fig_fc.update_traces(line_color=C_BLUE, fillcolor="rgba(16,122,226,0.07)", line_width=2)
    fig_fc.add_hline(y=80, line_dash="dot", line_color=C_GRAY, line_width=1,
                     annotation_text="80% do valor", annotation_font_size=8,
                     annotation_position="bottom right")
    fig_fc.update_layout(**plot_base(300))
    fig_fc.update_xaxes(showgrid=False, zeroline=False, tickangle=-40, tickfont_size=8)
    fig_fc.update_yaxes(showgrid=True, gridcolor="#f0f4f0", zeroline=False)
    fig_fc.update_layout(showlegend=False)
    st.plotly_chart(fig_fc, use_container_width=True)

# ═══════════════════════════════
# TEMPORAL
# ═══════════════════════════════
elif pagina == "Temporal":
    st.markdown("<div class='page-title'>Evolução Temporal</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Gastos e anomalias ao longo da 57ª Legislatura (2023–2026)</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        sec("Gasto Total por Ano (R$)")
        pa = df_f.groupby("ano")["valor_liquido"].sum().reset_index()
        fig_pa = px.bar(pa, x="ano", y="valor_liquido",
                        color="valor_liquido",
                        color_continuous_scale=[[0,"#dde8dd"],[1,C_BLUE]])
        fig_pa.update_traces(marker_line_width=0)
        fig_pa.update_layout(**plot_base(240))
        fig_pa.update_coloraxes(showscale=False)
        fig_pa.update_xaxes(showgrid=False, zeroline=False, title="Ano", type="category")
        fig_pa.update_yaxes(showgrid=True, gridcolor="#f0f4f0", zeroline=False, title="R$")
        st.plotly_chart(fig_pa, use_container_width=True)

    with col_r:
        sec("Anomalias por Ano")
        aa = anom_f.groupby("ano").agg(qtd=("valor_liquido","count"),val=("valor_liquido","sum")).reset_index()
        fig_aa = px.bar(aa, x="ano", y="qtd",
                        color="val", color_continuous_scale=[[0,"#fde8e8"],[1,C_RED]],
                        labels={"val":"Valor Total (R$)"})
        fig_aa.update_traces(marker_line_width=0)
        fig_aa.update_layout(**plot_base(240))
        fig_aa.update_coloraxes(showscale=False)
        fig_aa.update_xaxes(showgrid=False, zeroline=False, title="Ano", type="category")
        fig_aa.update_yaxes(showgrid=True, gridcolor="#f0f4f0", zeroline=False, title="Quantidade")
        st.plotly_chart(fig_aa, use_container_width=True)

    sec("Evolução Mensal — Volume Total e Anomalias")
    mt = df_f.groupby(["ano","month"])["valor_liquido"].sum().reset_index()
    mt["ano_mes"] = pd.to_datetime({"year":mt["ano"],"month":mt["month"],"day":1})
    ma = anom_f.groupby(["ano","month"])["valor_liquido"].agg(["count","sum"]).reset_index()
    ma.columns = ["ano","month","qtd_anom","val_anom"]
    ma["ano_mes"] = pd.to_datetime({"year":ma["ano"],"month":ma["month"],"day":1})

    fig_m = make_subplots(specs=[[{"secondary_y":True}]])
    fig_m.add_trace(go.Scatter(x=mt["ano_mes"], y=mt["valor_liquido"],
                               name="Total (R$)",
                               line=dict(color=C_BLUE,width=2),
                               fill="tozeroy",
                               fillcolor="rgba(16,122,226,0.06)"),
                    secondary_y=False)
    fig_m.add_trace(go.Bar(x=ma["ano_mes"], y=ma["qtd_anom"],
                           name="Qtd Anomalias",
                           marker_color=C_RED, opacity=0.65,
                           marker_line_width=0),
                    secondary_y=True)
    fig_m.update_layout(
        height=280, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", size=10, color="#444"),
        margin=dict(l=12,r=12,t=16,b=12),
        hovermode="x unified",
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.01),
        xaxis=dict(showgrid=False, zeroline=False),
    )
    fig_m.update_yaxes(title_text="Valor Total (R$)", secondary_y=False,
                       showgrid=True, gridcolor="#f0f4f0", zeroline=False,
                       title_font_size=9)
    fig_m.update_yaxes(title_text="Qtd Anomalias", secondary_y=True,
                       showgrid=False, zeroline=False, title_font_size=9)
    st.plotly_chart(fig_m, use_container_width=True)

    sec("Taxa de Anomalia Mensal (%)")
    mt2 = df_f.groupby(["ano","month"])["valor_liquido"].count().reset_index()
    mt2.columns = ["ano","month","total"]
    mt2["ano_mes"] = pd.to_datetime({"year":mt2["ano"],"month":mt2["month"],"day":1})
    tm = mt2.merge(ma[["ano_mes","qtd_anom"]], on="ano_mes", how="left").fillna(0)
    tm["taxa"] = tm["qtd_anom"] / tm["total"] * 100
    fig_t = px.area(tm, x="ano_mes", y="taxa",
                    labels={"ano_mes":"","taxa":"Taxa (%)"})
    fig_t.update_traces(line_color=C_RED, fillcolor="rgba(192,57,43,0.06)", line_width=2)
    fig_t.add_hline(y=5, line_dash="dot", line_color=C_GRAY, line_width=1,
                    annotation_text="threshold 5%", annotation_font_size=8,
                    annotation_position="top right")
    fig_t.update_layout(**plot_base(200))
    fig_t.update_xaxes(showgrid=False, zeroline=False)
    fig_t.update_yaxes(showgrid=True, gridcolor="#f0f4f0", zeroline=False)
    st.plotly_chart(fig_t, use_container_width=True)


# ═══════════════════════════════
# HIPÓTESES
# ═══════════════════════════════
elif pagina == "Hipóteses":

    st.markdown("<div class='page-title'>Validação das Hipóteses de Negócio</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Resultados da análise exploratória e machine learning</div>", unsafe_allow_html=True)

    sec("H1 • Categorias concentram os gastos")
    h1 = (df_f.groupby("tipo_despesa")["valor_liquido"]
            .sum().reset_index()
            .sort_values("valor_liquido", ascending=False))
    h1["perc"] = h1["valor_liquido"] / h1["valor_liquido"].sum() * 100

    fig_h1 = px.bar(h1.head(10), x="perc", y="tipo_despesa",
                    orientation="h", text="perc",
                    color="perc",
                    color_continuous_scale=[[0,"#dde8dd"],[1,C_BLUE]])
    fig_h1.update_layout(**plot_base(420))
    fig_h1.update_coloraxes(showscale=False)
    st.plotly_chart(fig_h1, use_container_width=True)

    sec("H2 • Gasto Médio com Passagens por UF")

    estados_distantes = [
        'AM', 'PA', 'RR', 'AP', 'AC', 'RO',
        'RS', 'SC', 'PR'
    ]

    aux2 = df_f[
        df_f['tipo_despesa'].str.contains('PASSAGEM', na=False)
    ].copy()

    aux2['regiao'] = aux2['uf'].apply(
        lambda x: 'Distante' if x in estados_distantes else 'Próximo'
    )

    aux2_grupo = (
        aux2.groupby(['uf', 'regiao'])['valor_liquido']
        .mean()
        .reset_index()
    )

    aux2_grupo = aux2_grupo.sort_values(
        'valor_liquido',
        ascending=False
    )

    fig_h2 = px.bar(
        aux2_grupo,
        x='uf',
        y='valor_liquido',
        color='regiao',
        barmode='group',
        labels={
            'uf': 'Estado',
            'valor_liquido': 'Gasto Médio (R$)',
            'regiao': 'Grupo'
        },
        color_discrete_map={
            'Distante': C_RED,
            'Próximo': C_BLUE
        }
    )

    fig_h2.update_traces(
        marker_line_width=0,
        hovertemplate=
        "<b>%{x}</b><br>" +
        "Gasto Médio: R$ %{y:,.2f}<br>" +
        "<extra></extra>"
    )

    fig_h2.update_layout(
        **plot_base(450),
        legend_title_text="",
        xaxis_title="Estado",
        yaxis_title="Gasto Médio (R$)"
    )

    fig_h2.update_xaxes(
        tickangle=-45,
        showgrid=False
    )

    fig_h2.update_yaxes(
        showgrid=True,
        gridcolor="#f0f4f0"
    )

    st.plotly_chart(
        fig_h2,
        use_container_width=True
    )

    sec("H4 • Evolução de gastos da legislatura")
    h4 = df_f.groupby("ano")["valor_liquido"].sum().reset_index()
    fig_h4 = px.line(h4, x="ano", y="valor_liquido", markers=True)
    fig_h4.update_layout(**plot_base(300))
    st.plotly_chart(fig_h4, use_container_width=True)

    sec("H5 • Concentração de fornecedores")
    forn = df_f.groupby("nome_fornecedor")["valor_liquido"].sum().sort_values(ascending=False)
    pct_top20 = forn.head(20).sum() / forn.sum() * 100
    st.metric("Top 20 fornecedores representam 21.2% do total gasto", f"{pct_top20:.1f}%")

    sec("H6 • Outliers Estatísticos (IQR)")

    q1 = df_f.groupby('tipo_despesa')['valor_liquido'].transform(
        lambda x: x.quantile(0.25)
    )

    q3 = df_f.groupby('tipo_despesa')['valor_liquido'].transform(
        lambda x: x.quantile(0.75)
    )

    limite_iqr = q3 + (3 * (q3 - q1))

    outliers_iqr = df_f[
        df_f['valor_liquido'] > limite_iqr
    ].copy()

    st.write(f"**Total de registros outliers:** {len(outliers_iqr):,}")

    st.write(
        f"**Percentual sobre o total:** "
        f"{len(outliers_iqr)/len(df_f)*100:.2f}%"
    )

    resumo_outliers = (
        outliers_iqr
        .groupby('tipo_despesa')['valor_liquido']
        .agg(['count', 'mean', 'max'])
        .sort_values('count', ascending=False)
    )

    st.dataframe(
        resumo_outliers,
        use_container_width=True
    )

    sec("H7 • Dispersão em Divulgação Parlamentar")

    div = df_f[df_f["tipo_despesa"] == "DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR."]

    top_partidos = div["partido"].value_counts().head(10).index

    fig_h7 = px.box(
        div[div["partido"].isin(top_partidos)],
        x="partido",
        y="valor_liquido"
    )

    fig_h7.update_layout(**plot_base(420))
    st.plotly_chart(fig_h7, use_container_width=True)
