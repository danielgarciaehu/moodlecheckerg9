"""
Moodle Detective – Panel de Control Forense
Detecta comportamientos sospechosos (velocidad excesiva) en logs de Moodle.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Página: DEBE ser el primer comando Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Moodle Detective",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
SESSION_GAP_MIN  = 45    # minutos de inactividad → nueva sesión
MAX_SESSION_MIN  = 180   # cap máximo por sesión (3 h)
CHUNK_ROWS       = 10_000

# Prefijos de "Contexto del evento" que NO son actividades de estudiante
SKIP_CTX = (
    "Curso:", "assign (id", "quiz (id", "page (id", "label (id",
    "pdfannotator (id", "h5pactivity (id", "scorm (id",
    "url (id", "folder (id", "resource (id", "book (id",
    "glossary (id", "wiki (id", "forum (id", "chat (id",
)

COMPLETION_KW = frozenset([
    "intento enviado",
    "se ha enviado una entrega",
    "entrega creada",
    "finalización de actividad de curso actualizada",
    "intento del cuestionario revisado",
    "módulo de curso visto",          # para recursos simples (auto-completado)
])


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def _trim_mean(series: pd.Series, alpha: float) -> float:
    """Media recortada (trimmed mean): descarta el α% inferior y superior."""
    arr = series.dropna().values.astype(float)
    if len(arr) == 0:
        return 0.0
    if alpha == 0 or len(arr) < 4:
        return float(arr.mean())
    k = int(np.floor(alpha * len(arr)))
    trimmed = np.sort(arr)[k: len(arr) - k]
    return float(trimmed.mean()) if len(trimmed) > 0 else float(arr.mean())


def _clean_ctx(ctx: str | None) -> str | None:
    """Extrae nombre limpio de actividad; devuelve None para eventos no válidos."""
    if not ctx or not isinstance(ctx, str):
        return None
    ctx = ctx.strip()
    if any(ctx.startswith(p) for p in SKIP_CTX):
        return None
    if re.search(r"\(id '[\d]+'\)\s*borrado", ctx):
        return None
    m = re.match(r'^[^:]+:\s*(.+)$', ctx)
    return (m.group(1).strip() if m else ctx) or None


def _fmt_min(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m"


# ─────────────────────────────────────────────────────────────────────────────
# Carga y parseo del CSV
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=3)
def parse_csv(raw: bytes) -> pd.DataFrame:
    """Lee el CSV en crudo y devuelve un DataFrame limpio (sin filtros de usuario)."""
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            chunks = []
            buf = io.BytesIO(raw)
            for chunk in pd.read_csv(buf, dtype=str, encoding=enc, chunksize=CHUNK_ROWS):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        st.error("No se pudo leer el CSV. Comprueba la codificación del fichero.")
        st.stop()

    df.columns = [c.strip().lstrip('﻿') for c in df.columns]
    rename = {
        "Hora":                          "ts_raw",
        "Nombre completo del usuario":   "user",
        "Usuario afectado":              "aff_user",
        "Contexto del evento":           "context",
        "Componente":                    "component",
        "Nombre evento":                 "event_name",
        "Descripción":                   "description",
        "Dirección IP":                  "ip",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["timestamp"] = pd.to_datetime(
        df["ts_raw"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["activity"]    = df["context"].apply(_clean_ctx)
    df["date"]        = df["timestamp"].dt.date
    df["event_lower"] = df["event_name"].str.lower().fillna("")
    return df


def filter_students(df: pd.DataFrame, exclude_names: list[str]) -> pd.DataFrame:
    """Elimina docentes, admins y filas de sistema."""
    excl = {n.upper().strip() for n in exclude_names if n.strip()}
    mask = (
        df["user"].notna()
        & (df["user"].str.strip() != "")
        & (df["user"].str.strip() != "-")
    )
    if excl:
        mask &= ~df["user"].str.upper().str.strip().isin(excl)
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Análisis de orden de actividades
# ─────────────────────────────────────────────────────────────────────────────
def infer_order(df: pd.DataFrame) -> list[str]:
    """
    Infiere el orden habitual de las actividades a partir del rango mediano
    de primera visita de todos los estudiantes.
    """
    sub = df[df["activity"].notna()]
    fv = sub.groupby(["user", "activity"])["timestamp"].min().reset_index()
    fv["rank"] = fv.groupby("user")["timestamp"].rank(method="first")
    return (
        fv.groupby("activity")["rank"]
        .median()
        .sort_values()
        .index.tolist()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de tiempos por (usuario, actividad)
# ─────────────────────────────────────────────────────────────────────────────
def activity_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (user, activity) calcula:
      total_min, n_events, completed, first_visit, last_event
    """
    sub = df[df["activity"].notna()].copy()
    records = []
    for (user, act), grp in sub.groupby(["user", "activity"]):
        grp = grp.sort_values("timestamp")
        ts  = grp["timestamp"].tolist()
        evs = grp["event_lower"].tolist()

        completed = any(kw in ev for ev in evs for kw in COMPLETION_KW)

        # Para cuestionarios: usar "comenzado el intento" → "intento enviado"
        has_start = grp["event_lower"].str.contains("comenzado el intento", na=False)
        has_end   = grp["event_lower"].str.contains("intento enviado",       na=False)
        if has_start.any() and has_end.any():
            starts = grp.loc[has_start, "timestamp"].values
            ends   = grp.loc[has_end,   "timestamp"].values
            total_min = 0.0
            for s in starts:
                future = ends[ends > s]
                if len(future):
                    total_min += min(
                        (future[0] - s) / np.timedelta64(1, "m"),
                        MAX_SESSION_MIN
                    )
        else:
            # Método general: sesiones con gap > SESSION_GAP_MIN
            if len(ts) <= 1:
                total_min = 0.0
            else:
                total_min  = 0.0
                sess_start = ts[0]
                prev       = ts[0]
                for t in ts[1:]:
                    gap = (t - prev).total_seconds() / 60
                    if gap > SESSION_GAP_MIN:
                        dur = (prev - sess_start).total_seconds() / 60
                        total_min += min(dur, MAX_SESSION_MIN)
                        sess_start = t
                    prev = t
                dur = (prev - sess_start).total_seconds() / 60
                total_min += min(dur, MAX_SESSION_MIN)

        records.append({
            "user":        user,
            "activity":    act,
            "total_min":   round(total_min, 1),
            "n_events":    len(ts),
            "completed":   completed,
            "first_visit": ts[0],
            "last_event":  ts[-1],
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Puntuación de sospecha
# ─────────────────────────────────────────────────────────────────────────────
def suspicion_scores(
    times: pd.DataFrame,
    alpha: float,
    threshold: float,
) -> pd.DataFrame:
    """
    Puntuación compuesta de sospecha (0-100) por estudiante basada en:
      1. Velocidad media por actividad (z-score respecto a media recortada)
      2. Sprint diario (máx. actividades en un solo día)
      3. Tiempo total vs media recortada del grupo
      4. Fracción de actividades realizadas por debajo del umbral de velocidad
    """
    if times.empty:
        return pd.DataFrame()

    # Estadísticas por actividad (media recortada)
    act_stats: dict[str, dict] = {}
    for act, grp in times.groupby("activity"):
        tm = _trim_mean(grp["total_min"], alpha)
        ts = float(grp["total_min"].std(ddof=1)) if len(grp) > 1 else 1.0
        act_stats[act] = {"tmean": max(tm, 0.01), "tstd": max(ts, 0.01)}

    # Actividades por día por usuario
    apd = (
        times.assign(date=times["first_visit"].dt.date)
        .groupby(["user", "date"])["activity"]
        .nunique()
    )
    max_apd = apd.groupby("user").max()

    # Tiempo total por usuario
    total_by_user  = times.groupby("user")["total_min"].sum()
    pop_trim_mean  = _trim_mean(total_by_user.reset_index()["total_min"], alpha)

    records = []
    for user, udf in times.groupby("user"):
        z_list, fast_acts = [], []

        for _, row in udf.iterrows():
            st_ = act_stats.get(row["activity"])
            if not st_ or st_["tmean"] < 0.5:   # actividades triviales (< 30 s media)
                continue
            z = (st_["tmean"] - row["total_min"]) / st_["tstd"]
            z_list.append(z)
            if row["total_min"] < st_["tmean"] * threshold:
                fast_acts.append(row["activity"])

        avg_z      = float(np.mean(z_list)) if z_list else 0.0
        max_day    = int(max_apd.get(user, 0))
        user_total = float(total_by_user.get(user, 0))
        time_ratio = user_total / pop_trim_mean if pop_trim_mean > 0 else 1.0

        first = udf["first_visit"].min()
        last  = udf["last_event"].max()
        days  = max((last - first).days, 0)

        # Fórmula compuesta (ponderada)
        score = min(100.0, max(0.0, (
            max(0.0, avg_z) * 30                          +   # velocidad
            max(0.0, (max_day - 4) / 6.0) * 20           +   # sprint diario
            max(0.0, 1.0 - time_ratio) * 35               +   # tiempo total
            (len(fast_acts) / max(len(udf), 1)) * 15          # fracción rápida
        )))

        records.append({
            "user":          user,
            "score":         round(score, 1),
            "total_min":     round(user_total, 1),
            "time_ratio":    round(time_ratio, 2),
            "avg_z":         round(avg_z, 2),
            "fast_n":        len(fast_acts),
            "fast_acts":     fast_acts,
            "max_apd":       max_day,
            "n_completed":   int(udf["completed"].sum()),
            "n_activities":  len(udf),
            "first_visit":   first,
            "last_visit":    last,
            "duration_days": days,
        })

    return (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS cosmético
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="metric-container"] { background:#f8f9fa; border-radius:8px; padding:10px 16px; }
  .alert-red    { color:#c62828; font-weight:700; }
  .alert-orange { color:#e65100; font-weight:700; }
  .alert-green  { color:#1b5e20; font-weight:700; }
  /* Pestañas compactas */
  [data-baseweb="tab-list"] {
      min-height: 0 !important;
      height: auto !important;
  }
  [data-baseweb="tab"] {
      padding: 3px 12px !important;
      min-height: 0 !important;
      height: auto !important;
      line-height: 1.4 !important;
      font-weight: normal !important;
  }
  button[data-baseweb="tab"] > div,
  button[data-baseweb="tab"] p {
      font-size: 0.82rem !important;
      padding: 0 !important;
      margin: 0 !important;
  }
  /* Pestaña activa: fondo granate tenue + negrita */
  [data-baseweb="tab"][aria-selected="true"] {
      background-color: rgba(120, 30, 30, 0.07) !important;
      border-radius: 4px 4px 0 0;
      font-weight: 700 !important;
  }
  [data-baseweb="tab"][aria-selected="true"] p {
      font-weight: 700 !important;
  }
</style>
<script>
(function() {
    var doc = (window.parent && window.parent.document) ? window.parent.document : document;

    // Cerrar sidebar solo la primera vez en esta sesión
    if (!sessionStorage.getItem('_sb_init')) {
        sessionStorage.setItem('_sb_init', '1');
        function tryClose() {
            var btn = doc.querySelector('[data-testid="collapsedControl"]');
            if (btn) { btn.click(); } else { setTimeout(tryClose, 150); }
        }
        setTimeout(tryClose, 250);
    }

    // Activar pestaña RANKING SOSPECHOSOS solo la primera vez
    if (!sessionStorage.getItem('_tab_init')) {
        sessionStorage.setItem('_tab_init', '1');
        function clickRanking() {
            var tabs = doc.querySelectorAll('[data-baseweb="tab"]');
            for (var t of tabs) {
                if (t.textContent.trim() === 'RANKING SOSPECHOSOS') { t.click(); return; }
            }
            setTimeout(clickRanking, 150);
        }
        setTimeout(clickRanking, 500);
    }
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Moodle Detective")
    st.caption("Análisis forense de tiempos")
    st.divider()

    st.subheader("Filtros de usuarios")
    teacher_input = st.text_area(
        "Excluir estos usuarios (uno por línea)",
        "DANIEL GARCIA GONZALEZ\nAdmin Usuario",
        height=110,
        help="Docentes y administradores que no son estudiantes",
    )
    exclude_list = [n.strip() for n in teacher_input.splitlines() if n.strip()]

    st.divider()
    st.subheader("Parámetros de análisis")

    alpha_pct = st.slider(
        "% de recorte para media robusta",
        min_value=5, max_value=25, value=10, step=5,
        help=(
            "**Media recortada** (*trimmed mean*): antes de calcular la media de "
            "referencia se descartan el X% más rápido Y el X% más lento. Así los "
            "casos extremos no distorsionan el patrón normal del grupo."
        ),
    )
    alpha = alpha_pct / 100

    threshold_pct = st.slider(
        "Umbral de velocidad sospechosa (%)",
        min_value=20, max_value=75, value=40, step=5,
        help=(
            "Una actividad se marca como sospechosa si el tiempo invertido es "
            "inferior a este % de la media recortada del grupo."
        ),
    )
    threshold = threshold_pct / 100

    st.divider()
    with st.expander("¿Qué es la media recortada?"):
        st.markdown(
            "En estadística, la **media recortada** (*trimmed mean* o *truncated mean*) "
            "es una medida robusta de tendencia central: se ordenan los valores, "
            "se descartan los extremos (aquí el α% inferior y α% superior) y se "
            "promedia el resto.\n\n"
            "Aquí sirve para evitar que los alumnos más rápidos (sospechosos) "
            "**arrastren la media hacia abajo**, lo que haría que el resto pareciera "
            "más lento de lo que realmente es."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Subida del fichero
# ─────────────────────────────────────────────────────────────────────────────
st.title("Detector de alumnos sprinters para eGela")
st.caption("Detección forense de comportamientos sospechosos en cursos Moodle")

uploaded = st.file_uploader(
    "Mostrando datos de ejemplo. Sube el 'log' de tu propio Moodle en CSV para analizarlo",
    type=["csv"],
    help="Moodle → Administración del curso → Informes → Registros → Descargar en formato CSV",
)

# ─────────────────────────────────────────────────────────────────────────────
# Procesamiento
# ─────────────────────────────────────────────────────────────────────────────
_ejemplo = Path(__file__).parent / "logs_ejemplo.csv"

if uploaded is not None:
    raw_bytes = uploaded.read()
    using_example = False
elif _ejemplo.exists():
    raw_bytes = _ejemplo.read_bytes()
    using_example = True
else:
    st.info(
        "Sube un fichero CSV de logs de Moodle para iniciar el análisis.\n\n"
        "**¿Cómo exportarlo?**  \n"
        "Moodle → Administración del curso → Informes → Registros → selecciona todo el periodo → "
        "botón *Obtener estos registros* → *Descargar en formato CSV*"
    )
    st.stop()

progress_bar = st.progress(0, text="Cargando CSV…")
with st.spinner(""):
    df_raw = parse_csv(raw_bytes)
    progress_bar.progress(30, text="Filtrando participantes…")
    df_full = filter_students(df_raw, exclude_list)
    progress_bar.progress(40, text="Preparando filtro de fechas…")
progress_bar.empty()

if df_full.empty:
    st.error("No se encontraron eventos de estudiantes en el CSV. Revisa los nombres a excluir.")
    st.stop()

# ── Selector de periodo ────────────────────────────────────────────────────────
_PERIODOS = [
    "Total histórico",
    "Últimos 12 meses",
    "Últimos 6 meses",
    "Último mes",
    "Última semana",
    "Rango personalizado",
]

_dmin_full = df_full["date"].min()
_dmax_full = df_full["date"].max()

_col_gap, col_periodo = st.columns([3, 1])
with col_periodo:
    periodo_sel = st.selectbox(
        "Periodo",
        _PERIODOS,
        index=0,
        label_visibility="collapsed",
        key="periodo_sel",
    )
    if periodo_sel == "Rango personalizado":
        _default_from = max(_dmin_full, _dmax_full - timedelta(days=30))
        _rango = st.date_input(
            "Rango",
            value=(_default_from, _dmax_full),
            min_value=_dmin_full,
            max_value=_dmax_full,
            label_visibility="collapsed",
            key="rango_custom",
        )
        if isinstance(_rango, (list, tuple)) and len(_rango) == 2:
            _f_from, _f_to = _rango[0], _rango[1]
        else:
            _f_from, _f_to = _dmin_full, _dmax_full
    else:
        _f_to = _dmax_full
        if periodo_sel == "Últimos 12 meses":
            _f_from = _dmax_full - timedelta(days=365)
        elif periodo_sel == "Últimos 6 meses":
            _f_from = _dmax_full - timedelta(days=182)
        elif periodo_sel == "Último mes":
            _f_from = _dmax_full - timedelta(days=30)
        elif periodo_sel == "Última semana":
            _f_from = _dmax_full - timedelta(days=7)
        else:  # Total histórico
            _f_from = _dmin_full

# Aplicar filtro de fechas
df = df_full[(df_full["date"] >= _f_from) & (df_full["date"] <= _f_to)].copy()

if df.empty:
    st.warning("No hay eventos en el periodo seleccionado. Amplía el rango de fechas.")
    st.stop()

progress_bar2 = st.progress(0, text="Infiriendo orden de actividades…")
with st.spinner(""):
    act_order = infer_order(df)
    progress_bar2.progress(40, text="Calculando tiempos por actividad…")

    t_df = activity_times(df)
    progress_bar2.progress(75, text="Calculando puntuaciones de sospecha…")

    sus_df = suspicion_scores(t_df, alpha, threshold)
    progress_bar2.progress(100, text="¡Listo!")
progress_bar2.empty()

if t_df.empty:
    st.error("No se encontraron eventos de estudiantes en el CSV. Revisa los nombres a excluir.")
    st.stop()

# Estadísticas globales
n_students   = df["user"].nunique()
n_acts       = len(act_order)
n_events     = len(df)
n_suspicious = int((sus_df["score"] >= 40).sum())
date_min     = df["date"].min()
date_max     = df["date"].max()
course_name  = ""
if "context" in df_raw.columns:
    curso_rows = df_raw[df_raw["context"].str.startswith("Curso:", na=False)]
    if not curso_rows.empty:
        ctx = curso_rows["context"].iloc[0]
        m = re.match(r'^Curso:\s*(.+)$', ctx)
        course_name = m.group(1).strip() if m else ""

if course_name:
    st.caption(f"Curso analizado: **{course_name}**")

# ─────────────────────────────────────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab_ranking, tab_general, tab_temporal, tab_tiempos, tab_individual = st.tabs([
    "RANKING SOSPECHOSOS",
    "PANEL GENERAL",
    "LINEA TEMPORAL",
    "TIEMPOS POR ACTIVIDAD",
    "INFORME INDIVIDUAL",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Panel General
# ═══════════════════════════════════════════════════════════════════════════════
with tab_general:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participantes",         n_students)
    c2.metric("Actividades",           n_acts)
    c3.metric("Eventos totales",       f"{n_events:,}")
    c4.metric("Sospechosos (>=40)",     n_suspicious)
    c5.metric("Dias analizados",       (date_max - date_min).days)

    st.divider()

    col_left, col_right = st.columns([3, 1])

    # ── Mapa de calor principal ───────────────────────────────────────────────
    with col_left:
        st.subheader("Mapa de calor: tiempo relativo (usuario × actividad)")

        # Calcular ratio = tiempo_usuario / media_recortada por actividad
        act_tm = {
            act: max(_trim_mean(grp["total_min"], alpha), 0.01)
            for act, grp in t_df.groupby("activity")
        }
        t_heat = t_df.copy()
        t_heat["ratio"] = t_heat.apply(
            lambda r: r["total_min"] / act_tm.get(r["activity"], 1.0), axis=1
        ).clip(0, 3)

        order_in_data = [a for a in act_order if a in t_heat["activity"].unique()]
        pivot = t_heat.pivot_table(
            index="user", columns="activity", values="ratio", aggfunc="mean"
        ).reindex(columns=order_in_data)

        # Ordenar filas por puntuación de sospecha (mayor sospecha arriba)
        user_ord = sus_df["user"].tolist()
        pivot    = pivot.reindex([u for u in user_ord if u in pivot.index])

        x_labels = [a[:32] + "…" if len(a) > 32 else a for a in order_in_data]

        fig_heat = px.imshow(
            pivot.values,
            x=x_labels,
            y=pivot.index.tolist(),
            color_continuous_scale=[
                [0.00, "#b71c1c"],
                [0.20, "#ef5350"],
                [0.40, "#ffeb3b"],
                [0.70, "#aed581"],
                [1.00, "#2e7d32"],
            ],
            zmin=0, zmax=3,
            aspect="auto",
            labels={"color": "Ratio vs media"},
        )
        fig_heat.update_layout(
            height=max(320, n_students * 24 + 160),
            xaxis_tickangle=-40,
            margin=dict(l=160, r=20, t=10, b=150),
            coloraxis_colorbar=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.22, yanchor="top",
                thickness=14, len=0.7,
                title=dict(text="Ratio respecto a la media del grupo", side="top"),
                tickvals=[0, 0.4, 1, 2, 3],
                ticktext=["0 — muy rapido", "0.4 — umbral", "1 — media", "2", "3 — lento"],
            ),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Panel derecho ─────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Top sospechosos")
        top_n = min(10, len(sus_df))
        top   = sus_df.head(top_n)[["user", "score"]].copy()
        top["Nivel"] = top["score"].apply(
            lambda s: "ALTO" if s >= 60 else ("MEDIO" if s >= 40 else "OK")
        )
        top = top.rename(columns={"user": "Participante", "score": "Punt."})
        st.dataframe(top[["Nivel", "Participante", "Punt."]], hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Usuarios activos / día")
        daily = (
            df.groupby("date")["user"]
            .nunique()
            .reset_index()
            .rename(columns={"user": "n"})
        )
        fig_daily = px.bar(
            daily, x="date", y="n",
            color_discrete_sequence=["#1976d2"],
            labels={"date": "", "n": "Usuarios"},
        )
        fig_daily.update_layout(height=180, margin=dict(t=0, b=30, l=30, r=10))
        st.plotly_chart(fig_daily, use_container_width=True)

        st.divider()
        st.subheader("Actividades completadas / estudiante")
        comp = t_df.groupby("user")["completed"].sum().reset_index()
        fig_comp = px.histogram(
            comp, x="completed", nbins=20,
            color_discrete_sequence=["#43a047"],
            labels={"completed": "Act. completadas", "count": "Estudiantes"},
        )
        fig_comp.update_layout(height=180, margin=dict(t=0, b=30, l=30, r=10))
        st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Línea Temporal
# ═══════════════════════════════════════════════════════════════════════════════
with tab_temporal:
    st.subheader("Linea temporal del curso")

    # Controles de la vista temporal
    c_range, c_days = st.columns([2, 1])
    with c_range:
        date_range = st.date_input(
            "Rango de fechas",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
    with c_days:
        rush_threshold = st.number_input(
            "Sprint: actividades/día para marcar en rojo",
            min_value=3, max_value=20, value=6,
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = date_min, date_max

    n_days    = (d_end - d_start).days + 1
    all_dates = [d_start + timedelta(days=i) for i in range(n_days)]

    # Actividades por día por usuario en ese rango
    apd_full = (
        t_df.assign(date=t_df["first_visit"].dt.date)
        .query("@d_start <= date <= @d_end")
        .groupby(["user", "date"])
        .agg(n_acts=("activity", "nunique"), total_min=("total_min", "sum"))
        .reset_index()
    )

    user_ord_tl = sus_df["user"].tolist()
    matrix_acts = np.full((len(user_ord_tl), n_days), np.nan)
    matrix_mins = np.full((len(user_ord_tl), n_days), np.nan)

    for _, row in apd_full.iterrows():
        if row["user"] in user_ord_tl:
            ui = user_ord_tl.index(row["user"])
            di = (row["date"] - d_start).days
            if 0 <= di < n_days:
                matrix_acts[ui, di] = row["n_acts"]
                matrix_mins[ui, di] = row["total_min"]

    x_labels_tl = [str(d) for d in all_dates]
    y_labels_tl = user_ord_tl

    tab2a, tab2b = st.tabs(["Actividades por día", "Minutos por día"])

    with tab2a:
        st.caption("Cada celda = actividades distintas completadas ese día.")
        fig_tl1 = px.imshow(
            matrix_acts,
            x=x_labels_tl, y=y_labels_tl,
            color_continuous_scale=[
                [0.00, "#ffffff"],
                [0.01, "#e3f2fd"],
                [0.40, "#1565c0"],
                [0.70, "#ff6f00"],
                [1.00, "#b71c1c"],
            ],
            zmin=0, zmax=max(rush_threshold, 1),
            aspect="auto",
            labels={"color": "Act./día"},
        )
        fig_tl1.update_layout(
            height=max(320, len(user_ord_tl) * 24 + 160),
            xaxis_tickangle=-45,
            margin=dict(l=160, r=20, t=10, b=150),
            coloraxis_colorbar=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.22, yanchor="top",
                thickness=14, len=0.7,
                title=dict(text="Actividades por día (rojo = sprint)", side="top"),
                tickvals=[0, round(rush_threshold * 0.4), round(rush_threshold * 0.7), rush_threshold],
                ticktext=["0 — sin actividad", f"{round(rush_threshold*0.4)} — ritmo normal",
                          f"{round(rush_threshold*0.7)}", f"{rush_threshold} — sprint"],
            ),
        )
        st.plotly_chart(fig_tl1, use_container_width=True)

    with tab2b:
        valid_mins = matrix_mins[matrix_mins > 0]
        avg_day    = float(valid_mins.mean()) if valid_mins.size else 30
        st.caption("Cada celda = minutos totales dedicados al curso ese día.")
        fig_tl2 = px.imshow(
            matrix_mins,
            x=x_labels_tl, y=y_labels_tl,
            color_continuous_scale=[
                [0.00, "#ffffff"],
                [0.01, "#e8f5e9"],
                [0.50, "#2e7d32"],
                [0.80, "#ff8f00"],
                [1.00, "#b71c1c"],
            ],
            zmin=0, zmax=avg_day * 3,
            aspect="auto",
            labels={"color": "Min./día"},
        )
        fig_tl2.update_layout(
            height=max(320, len(user_ord_tl) * 24 + 160),
            xaxis_tickangle=-45,
            margin=dict(l=160, r=20, t=10, b=150),
            coloraxis_colorbar=dict(
                orientation="h",
                x=0.5, xanchor="center",
                y=-0.22, yanchor="top",
                thickness=14, len=0.7,
                title=dict(text="Minutos por día (blanco = inactividad, verde = normal, rojo = muy poco)", side="top"),
                tickvals=[0, round(avg_day), round(avg_day * 2), round(avg_day * 3)],
                ticktext=["0", f"{round(avg_day)} min (media)", f"{round(avg_day*2)}", f"{round(avg_day*3)}+"],
            ),
        )
        st.plotly_chart(fig_tl2, use_container_width=True)

    st.caption(
        "Los participantes están ordenados por puntuación de sospecha (mayor sospecha arriba). "
        "Los más sospechosos tienden a concentrar actividades en muy pocos días."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Tiempos por Actividad
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tiempos:
    st.subheader("Distribución de tiempos por actividad")

    # Enriquecer t_df con estadísticas
    act_tm3 = {
        act: max(_trim_mean(grp["total_min"], alpha), 0.01)
        for act, grp in t_df.groupby("activity")
    }
    t_df3 = t_df[t_df["activity"].isin(act_order)].copy()
    t_df3["act_idx"]    = t_df3["activity"].map({a: i for i, a in enumerate(act_order)})
    t_df3               = t_df3.sort_values("act_idx")
    t_df3["tmean"]      = t_df3["activity"].map(act_tm3)
    t_df3["ratio"]      = (t_df3["total_min"] / t_df3["tmean"]).round(2)
    t_df3["suspicious"] = t_df3["total_min"] < t_df3["tmean"] * threshold
    t_df3["cat"]        = t_df3["suspicious"].map({True: "Sospechoso", False: "Normal"})
    t_df3["act_short"]  = t_df3["activity"].apply(lambda a: a[:35] + "…" if len(a) > 35 else a)

    short_order = [
        a[:35] + "…" if len(a) > 35 else a
        for a in act_order
        if a in t_df3["activity"].unique()
    ]

    sub3a, sub3b, sub3c = st.tabs(["Boxplot global", "Barras por actividad", "Tabla resumen"])

    # ── Boxplot ───────────────────────────────────────────────────────────────
    with sub3a:
        st.caption("Distribución de tiempos de todos los participantes por actividad. Puntos rojos = sospechosos.")
        fig_box = px.box(
            t_df3,
            x="act_short", y="total_min",
            color="cat",
            color_discrete_map={"Sospechoso": "#d32f2f", "Normal": "#2e7d32"},
            points="all",
            hover_data=["user", "total_min", "n_events"],
            labels={"act_short": "", "total_min": "Minutos", "cat": ""},
            category_orders={"act_short": short_order},
        )
        fig_box.update_layout(
            height=max(560, len(act_order) * 22 + 200),
            xaxis_tickangle=-45,
            margin=dict(b=160), legend_title_text="",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Barras horizontales por actividad seleccionada ────────────────────────
    with sub3b:
        sel_act = st.selectbox("Selecciona actividad", options=act_order, key="act_sel3b")
        act_data = t_df3[t_df3["activity"] == sel_act].sort_values("total_min").copy()
        tmean_sel = act_tm3.get(sel_act, 0)

        act_data["bar_color"] = act_data["suspicious"].map(
            {True: "#d32f2f", False: "#2e7d32"}
        )

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=act_data["user"],
            x=act_data["total_min"],
            orientation="h",
            marker_color=act_data["bar_color"],
            text=act_data["total_min"].apply(lambda m: f"{m:.0f} min"),
            textposition="outside",
        ))
        fig_bar.add_vline(
            x=tmean_sel, line_dash="dash", line_color="#1565c0",
            annotation_text=f"Media recortada: {tmean_sel:.0f} min",
            annotation_position="top right",
        )
        fig_bar.add_vline(
            x=tmean_sel * threshold, line_dash="dot", line_color="#e65100",
            annotation_text=f"Umbral sospecha: {tmean_sel * threshold:.0f} min",
            annotation_position="bottom right",
        )
        fig_bar.update_layout(
            height=max(320, len(act_data) * 26 + 80),
            xaxis_title="Minutos",
            margin=dict(l=200, r=80),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    with sub3c:
        summary = []
        for act in act_order:
            grp = t_df3[t_df3["activity"] == act]
            if grp.empty:
                continue
            tm  = act_tm3.get(act, 0)
            sus = int(grp["suspicious"].sum())
            summary.append({
                "Actividad":           act,
                "Media recortada (m)": round(tm, 1),
                "Mediana (m)":         round(grp["total_min"].median(), 1),
                "Mín (m)":             round(grp["total_min"].min(), 1),
                "Participantes":       len(grp),
                "Sospechosos":         sus,
                "% Sosp.":             round(100 * sus / max(len(grp), 1), 0),
            })

        def _hl_pct(val):
            if val >= 50: return "background-color:#ffcdd2"
            if val >= 25: return "background-color:#fff9c4"
            return ""

        sum_df = pd.DataFrame(summary)
        st.dataframe(
            sum_df.style.map(_hl_pct, subset=["% Sosp."]),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Ranking Sospechosos
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ranking:
    st.subheader("Ranking de participantes sospechosos")

    with st.expander("Metodología de la puntuación", expanded=False):
        st.markdown(f"""
La puntuación de sospecha (0–100) combina **cuatro factores** ponderados:

| Factor | Peso | Descripción |
|--------|------|-------------|
| Velocidad media | 30 % | Z-score respecto a la media recortada por actividad |
| Sprint diario   | 20 % | Actividades completadas en un solo día (penaliza si > 4/día) |
| Tiempo total    | 35 % | Tiempo total invertido en el curso vs media recortada del grupo |
| Act. rápidas    | 15 % | Fracción de actividades por debajo del umbral ({threshold_pct}% de la media) |

**Media recortada usada:** descartando el {alpha_pct}% más rápido y el {alpha_pct}% más lento del grupo.
Esto evita que los propios sospechosos arrastren la media de referencia hacia abajo (*Winsorización*).
        """)

    col4a, col4b = st.columns([3, 2])

    with col4a:
        st.markdown("#### Tiempo total vs Sprint diario")
        fig_sc = px.scatter(
            sus_df,
            x="total_min", y="max_apd",
            color="score",
            size="n_activities",
            hover_name="user",
            hover_data=["score", "fast_n", "duration_days", "time_ratio"],
            color_continuous_scale=[[0, "#4caf50"], [0.4, "#ffeb3b"], [1, "#d32f2f"]],
            range_color=[0, 100],
            labels={
                "total_min": "Minutos totales en el curso",
                "max_apd":   "Máx. actividades en un solo día",
                "score":     "Puntuación",
            },
            text="user",
        )
        fig_sc.update_traces(textposition="top center", textfont_size=8)
        fig_sc.update_layout(height=420, margin=dict(t=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col4b:
        # Histograma de puntuaciones
        fig_hist = px.histogram(
            sus_df, x="score", nbins=20,
            color_discrete_sequence=["#1976d2"],
            labels={"score": "Puntuación de sospecha", "count": "Participantes"},
            title="Distribución de puntuaciones",
        )
        fig_hist.add_vline(x=40, line_dash="dash", line_color="#e65100",
                           annotation_text="Umbral medio (40)")
        fig_hist.add_vline(x=60, line_dash="dash", line_color="#d32f2f",
                           annotation_text="Umbral alto (60)")
        fig_hist.update_layout(height=220, title_font_size=13, margin=dict(t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Gráfico de araña del participante más sospechoso
        if len(sus_df):
            top    = sus_df.iloc[0]
            cats   = ["Velocidad", "Sprint diario", "Tiempo total", "Act. rápidas"]
            vals   = [
                round(max(0, top["avg_z"]) * 30, 1),
                round(max(0, (top["max_apd"] - 4) / 6) * 20, 1),
                round(max(0, 1.0 - top["time_ratio"]) * 35, 1),
                round((top["fast_n"] / max(top["n_activities"], 1)) * 15, 1),
            ]
            first_name = top["user"].split()[0] if top["user"] else "?"
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                line_color="#d32f2f",
                fillcolor="rgba(211,47,47,0.2)",
                name=first_name,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 35])),
                title=dict(text=f"Perfil: {first_name}", font_size=13),
                height=240, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Tabla completa de ranking
    st.markdown("#### Tabla completa")
    disp = sus_df.copy()
    disp["Nivel"] = disp["score"].apply(
        lambda s: "ALTO" if s >= 60 else ("MEDIO" if s >= 40 else "OK")
    )
    disp["Tiempo total"] = disp["total_min"].apply(_fmt_min)
    disp["fast_acts_str"] = disp["fast_acts"].apply(
        lambda lst: ", ".join(lst[:3]) + ("…" if len(lst) > 3 else "") if lst else "—"
    )
    cols_show = {
        "Nivel":         "Alerta",
        "user":          "Participante",
        "score":         "Puntuación",
        "Tiempo total":  "Tiempo total",
        "time_ratio":    "Ratio vs grupo",
        "max_apd":       "Máx act/día",
        "fast_n":        "Act. rápidas",
        "duration_days": "Días en curso",
        "n_completed":   "Act. completadas",
        "fast_acts_str": "Actividades sospechosas",
    }
    disp_show = disp[list(cols_show.keys())].rename(columns=cols_show)
    st.dataframe(disp_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Informe Individual
# ═══════════════════════════════════════════════════════════════════════════════
with tab_individual:
    st.subheader("Informe individual detallado")

    # ── Buscador + ordenación ─────────────────────────────────────────────────
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        txt_filter = st.text_input(
            "Buscar", placeholder="Escribe nombre o apellido…",
            label_visibility="collapsed", key="ind_search"
        )
    with fc2:
        sort_by = st.selectbox(
            "Orden", options=[
                "Por sospecha",
                "Alfabético (nombre)",
                "Alfabético (apellido)",
                "Último acceso (más reciente)",
                "Último acceso (más antiguo)",
            ],
            label_visibility="collapsed", key="ind_sort"
        )

    # Construir lista filtrada y ordenada
    candidates = sus_df.copy()
    if txt_filter.strip():
        mask = candidates["user"].str.contains(txt_filter.strip(), case=False, na=False)
        candidates = candidates[mask]

    if sort_by == "Alfabético (nombre)":
        candidates = candidates.sort_values("user")
    elif sort_by == "Alfabético (apellido)":
        # Ordenar por el segundo token (primer apellido)
        candidates = candidates.assign(
            _sort=candidates["user"].str.split().str[1:].str.join(" ")
        ).sort_values("_sort").drop(columns="_sort")
    elif sort_by == "Último acceso (más reciente)":
        candidates = candidates.sort_values("last_visit", ascending=False)
    elif sort_by == "Último acceso (más antiguo)":
        candidates = candidates.sort_values("last_visit", ascending=True)
    # "Por sospecha" ya viene ordenado por defecto

    student_opts = candidates["user"].tolist()

    if not student_opts:
        st.warning("Ningún participante coincide con la búsqueda.")
        st.stop()

    sel_user = st.selectbox(
        "Participante",
        options=student_opts,
        label_visibility="collapsed",
        key="sel_user_tab5",
    )

    if not sel_user:
        st.info("Selecciona un participante.")
        st.stop()

    sus_row   = sus_df[sus_df["user"] == sel_user].iloc[0]
    user_t    = t_df[t_df["user"] == sel_user].copy()
    score     = sus_row["score"]
    alert_str = "ALTO RIESGO" if score >= 60 else ("RIESGO MEDIO" if score >= 40 else "SIN RIESGO APARENTE")
    clr       = "#c62828" if score >= 60 else ("#e65100" if score >= 40 else "#1b5e20")

    st.markdown(
        f"**Puntuación de sospecha:** "
        f"<span style='color:{clr};font-size:1.5em;font-weight:700'>"
        f"{score:.0f} / 100 — {alert_str}</span>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tiempo total",        _fmt_min(sus_row["total_min"]))
    m2.metric("Ratio vs grupo",      f"{sus_row['time_ratio']:.0%}")
    m3.metric("Máx act/día",         sus_row["max_apd"])
    m4.metric("Act. rápidas",        sus_row["fast_n"])
    m5.metric("Días en el curso",    sus_row["duration_days"])
    m6.metric("Act. completadas",    sus_row["n_completed"])

    st.divider()

    col5a, col5b = st.columns(2)

    # ── Barras: usuario vs media recortada del grupo ──────────────────────────
    with col5a:
        st.markdown("#### Tiempo por actividad vs media del grupo")

        act_tm5 = {
            act: max(_trim_mean(grp["total_min"], alpha), 0.01)
            for act, grp in t_df.groupby("activity")
        }
        u5 = user_t[user_t["activity"].isin(act_order)].copy()
        u5["tmean"]    = u5["activity"].map(act_tm5)
        u5["ratio"]    = u5["total_min"] / u5["tmean"].clip(lower=0.01)
        u5["bar_color"] = u5["ratio"].apply(
            lambda r: "#d32f2f" if r < threshold else ("#ff9800" if r < 0.7 else "#2e7d32")
        )
        u5["act_idx"] = u5["activity"].map({a: i for i, a in enumerate(act_order)})
        u5 = u5.sort_values("act_idx")
        u5["act_label"] = u5["activity"].apply(lambda a: a[:42] + "…" if len(a) > 42 else a)

        fig5a = go.Figure()
        fig5a.add_trace(go.Bar(
            y=u5["act_label"], x=u5["total_min"],
            orientation="h",
            marker_color=u5["bar_color"],
            name="Este participante",
            text=u5["total_min"].apply(lambda m: f"{m:.0f} min"),
            textposition="outside",
        ))
        fig5a.add_trace(go.Scatter(
            y=u5["act_label"], x=u5["tmean"],
            mode="markers",
            marker=dict(color="#1565c0", size=9, symbol="line-ns-open", line_width=2),
            name="Media recortada del grupo",
        ))
        fig5a.update_layout(
            height=max(320, len(u5) * 28 + 60),
            xaxis_title="Minutos",
            legend=dict(orientation="h", y=1.04),
            margin=dict(l=20, r=80),
        )
        st.plotly_chart(fig5a, use_container_width=True)

    # ── Gantt de accesos ──────────────────────────────────────────────────────
    with col5b:
        st.markdown("#### Línea de tiempo de accesos (Gantt)")
        gantt_rows = []
        for _, row in user_t.iterrows():
            if pd.isna(row["activity"]):
                continue
            dur = max(row["total_min"], 1)
            gantt_rows.append({
                "Actividad": row["activity"][:42] + "…" if len(row["activity"]) > 42 else row["activity"],
                "Inicio":    row["first_visit"],
                "Fin":       row["first_visit"] + timedelta(minutes=dur),
                "Min":       row["total_min"],
            })
        if gantt_rows:
            gantt_df = pd.DataFrame(gantt_rows).sort_values("Inicio")
            fig_gantt = px.timeline(
                gantt_df,
                x_start="Inicio", x_end="Fin", y="Actividad",
                color="Min",
                color_continuous_scale=[[0, "#d32f2f"], [0.3, "#ff9800"], [1, "#2e7d32"]],
                labels={"Min": "Minutos"},
            )
            max_min = gantt_df["Min"].max() if not gantt_df.empty else 60
            fig_gantt.update_layout(
                height=max(380, len(gantt_rows) * 34 + 160),
                margin=dict(l=20, b=140),
                coloraxis_colorbar=dict(
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=-0.38, yanchor="top",
                    thickness=14, len=0.6,
                    title=dict(text="Minutos invertidos (rojo = poco, verde = suficiente)", side="top"),
                    tickvals=[0, round(max_min * 0.3), round(max_min * 0.7), round(max_min)],
                    ticktext=["0", f"{round(max_min*0.3)} min", f"{round(max_min*0.7)} min",
                              f"{round(max_min)} min"],
                ),
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

    st.divider()

    # ── Tabla detalle ─────────────────────────────────────────────────────────
    st.markdown("#### Detalle de actividades")

    u_detail = user_t[user_t["activity"].isin(act_order)].copy()
    u_detail["Media grupo (m)"] = u_detail["activity"].map(act_tm5).round(1)
    u_detail["Ratio"]           = (
        u_detail["total_min"] / u_detail["Media grupo (m)"].clip(lower=0.01)
    ).round(2)
    u_detail["Estado"] = u_detail["Ratio"].apply(
        lambda r: "Sospechoso" if r < threshold else ("Algo rapido" if r < 0.7 else "Normal")
    )
    u_detail["Completada"]  = u_detail["completed"].map({True: "Si", False: "No"})
    u_detail["act_idx"]     = u_detail["activity"].map({a: i for i, a in enumerate(act_order)})
    u_detail                = u_detail.sort_values("act_idx")
    u_detail["Fecha inicio"] = u_detail["first_visit"].dt.strftime("%d/%m/%y %H:%M")

    det_show = u_detail[[
        "Estado", "activity", "total_min", "Media grupo (m)",
        "Ratio", "n_events", "Completada", "Fecha inicio",
    ]].rename(columns={
        "activity":      "Actividad",
        "total_min":     "Tiempo (m)",
        "n_events":      "Eventos",
    })
    st.dataframe(det_show, use_container_width=True, hide_index=True)

    st.divider()

    # ── Exportar informe HTML ─────────────────────────────────────────────────
    st.markdown("#### Exportar informe")

    det_html = det_show.to_html(index=False, border=0, classes="det")
    score_color = "#c62828" if score >= 60 else ("#e65100" if score >= 40 else "#1b5e20")

    rows_meta = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in [
            ("Tiempo total en el curso",                _fmt_min(sus_row["total_min"])),
            ("Ratio vs media del grupo",                f"{sus_row['time_ratio']:.0%}"),
            ("Máx. actividades en un día",              sus_row["max_apd"]),
            ("Actividades realizadas por debajo del umbral de velocidad", sus_row["fast_n"]),
            ("Días totales en el curso",                sus_row["duration_days"]),
            ("Actividades completadas",                 sus_row["n_completed"]),
        ]
    )

    html_report = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Moodle Detective – {sel_user}</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; max-width:960px; margin:0 auto; padding:24px; color:#212121; }}
  h1   {{ color:#1a237e; border-bottom:3px solid #3f51b5; padding-bottom:8px; }}
  h2   {{ color:#283593; margin-top:32px; }}
  .score {{ font-size:2.2em; font-weight:700; color:{score_color}; }}
  table {{ border-collapse:collapse; width:100%; margin-bottom:20px; font-size:0.9em; }}
  th    {{ background:#3f51b5; color:#fff; padding:8px 10px; text-align:left; }}
  td    {{ padding:6px 10px; border-bottom:1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background:#f5f7ff; }}
  .det td:first-child {{ font-weight:500; }}
  .meta {{ font-size:0.8em; color:#757575; margin-top:32px; border-top:1px solid #e0e0e0; padding-top:12px; }}
  @media print {{ body {{ padding:0; }} }}
</style>
</head>
<body>
<h1>Moodle Detective — Informe de {sel_user}</h1>
<p style="color:#555">
  Curso: <strong>{course_name or "—"}</strong> &nbsp;|&nbsp;
  Generado: <strong>{datetime.now().strftime("%d/%m/%Y %H:%M")}</strong>
</p>

<h2>Resumen ejecutivo</h2>
<p>Puntuación de sospecha: <span class="score">{score:.0f} / 100</span>
   &nbsp;—&nbsp; {alert_str}</p>
<table>
  <tr><th>Métrica</th><th>Valor</th></tr>
  {rows_meta}
</table>

<h2>Detalle por actividad</h2>
{det_html}

<div class="meta">
  <strong>Metodología:</strong> Puntuación basada en media recortada (trimmed mean, α={alpha_pct}%).
  Umbral de velocidad sospechosa: {threshold_pct}% de la media recortada del grupo.<br>
  Generado con <em>Moodle Detective</em>.
</div>
</body>
</html>"""

    st.download_button(
        label="Descargar informe HTML (para imprimir / adjuntar a expediente)",
        data=html_report.encode("utf-8"),
        file_name=f"informe_{sel_user.replace(' ', '_').replace('/', '-')}.html",
        mime="text/html",
    )
