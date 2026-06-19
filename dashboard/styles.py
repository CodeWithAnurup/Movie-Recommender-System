"""Custom styles, theming, and HTML component generators."""

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────
CHART_COLORS = [
    "#1a73e8", "#ea4335", "#fbbc04", "#34a853", "#ff6d01",
    "#46bdc6", "#7baaf7", "#f07b72", "#a8dab5", "#e8710a",
    "#9334e6", "#f439a0", "#24c1e0", "#188038", "#b31412",
]

LIGHT = {
    "bg": "#ffffff",
    "bg2": "#f8f9fa",
    "card_bg": "#ffffff",
    "text": "#1f1f1f",
    "text2": "#5f6368",
    "border": "#e0e0e0",
    "accent": "#1a73e8",
    "accent_light": "#e8f0fe",
    "success": "#0d904f",
    "warning": "#e37400",
    "danger": "#c5221f",
    "plotly_template": "plotly_white",
    "plotly_grid": "#f0f0f0",
}

DARK = {
    "bg": "#0e1117",
    "bg2": "#1a1d23",
    "card_bg": "#1a1d23",
    "text": "#e8eaed",
    "text2": "#9aa0a6",
    "border": "#2d3136",
    "accent": "#8ab4f8",
    "accent_light": "#1c2a3e",
    "success": "#81c995",
    "warning": "#fdd663",
    "danger": "#f28b82",
    "plotly_template": "plotly_dark",
    "plotly_grid": "#2d3136",
}


def get_theme(dark_mode: bool = False) -> dict:
    """Return the active color theme dictionary."""
    return DARK if dark_mode else LIGHT


def get_plotly_layout(dark_mode: bool = False) -> dict:
    """Return a dict suitable for fig.update_layout(**layout)."""
    t = get_theme(dark_mode)
    return dict(
        template=t["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=t["text"], size=12),
        margin=dict(l=40, r=20, t=44, b=40),
        hoverlabel=dict(bgcolor=t["card_bg"], font_size=12, font_color=t["text"]),
        legend=dict(font=dict(size=11)),
        xaxis=dict(gridcolor=t["plotly_grid"], linecolor=t["border"]),
        yaxis=dict(gridcolor=t["plotly_grid"], linecolor=t["border"]),
        colorway=CHART_COLORS,
    )


# ── CSS Injection ────────────────────────────────────────────────
def inject_css(dark_mode: bool = False):
    """Inject global CSS into the Streamlit app."""
    t = get_theme(dark_mode)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global ─────────────────────────────────── */
        html, body, .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .stApp h1 {{ font-size: 1.45rem !important; font-weight: 700 !important; margin-bottom: 0.4rem !important; }}
        .stApp h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; margin-bottom: 0.35rem !important; }}
        .stApp h3 {{ font-size: 0.98rem !important; font-weight: 600 !important; margin-bottom: 0.25rem !important; }}

        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 1rem !important;
            max-width: 1200px;
        }}

        /* reduce vertical gaps everywhere */
        div[data-testid="stVerticalBlock"] > div {{
            gap: 0.35rem !important;
        }}
        div[data-testid="stHorizontalBlock"] {{
            gap: 0.6rem !important;
        }}

        /* ── Sidebar ────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            width: 250px !important;
            min-width: 250px !important;
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            font-size: 0.82rem !important;
        }}
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {{
            padding: 0.3rem 0 !important;
        }}

        /* ── Tabs ───────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.82rem; padding: 0.35rem 0.8rem; font-weight: 500;
        }}

        /* ── Plotly charts ──────────────────────────── */
        .stPlotlyChart {{ margin-bottom: 0.3rem !important; }}

        /* ── Metric Cards ───────────────────────────── */
        .metric-card {{
            background: {t['card_bg']};
            border: 1px solid {t['border']};
            border-radius: 10px;
            padding: 0.85rem 0.7rem;
            text-align: center;
            height: 100%;
            box-sizing: border-box;
            transition: box-shadow 0.2s ease;
        }}
        .metric-card:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .metric-label {{
            font-size: 0.68rem;
            font-weight: 600;
            color: {t['text2']};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
        }}
        .metric-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {t['accent']};
            line-height: 1.2;
        }}
        .metric-sub {{
            font-size: 0.72rem;
            color: {t['text2']};
            margin-top: 0.15rem;
        }}

        /* ── Recommendation Cards ───────────────────── */
        .rec-card {{
            background: {t['card_bg']};
            border: 1px solid {t['border']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
            transition: border-color 0.2s ease;
        }}
        .rec-card:hover {{
            border-color: {t['accent']};
        }}
        .rec-rank {{
            display: inline-block;
            background: {t['accent']};
            color: #fff;
            font-size: 0.7rem;
            font-weight: 700;
            border-radius: 4px;
            padding: 1px 7px;
            margin-right: 0.4rem;
            vertical-align: middle;
        }}
        .rec-title {{
            font-weight: 600;
            font-size: 0.92rem;
            color: {t['text']};
            margin-bottom: 0.25rem;
        }}
        .rec-meta {{
            font-size: 0.78rem;
            color: {t['text2']};
        }}
        .rec-score {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {t['accent']};
        }}

        /* ── Insight Boxes ──────────────────────────── */
        .insight-box {{
            background: {t['accent_light']};
            border-left: 3px solid {t['accent']};
            padding: 0.55rem 0.9rem;
            border-radius: 0 6px 6px 0;
            margin: 0.4rem 0;
            font-size: 0.82rem;
            color: {t['text']};
            line-height: 1.55;
        }}

        /* ── Business Cards ─────────────────────────── */
        .biz-card {{
            background: {t['card_bg']};
            border: 1px solid {t['border']};
            border-radius: 10px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.5rem;
            transition: box-shadow 0.2s ease;
        }}
        .biz-card:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .biz-card h4 {{
            margin: 0 0 0.35rem 0;
            font-size: 0.92rem;
            font-weight: 600;
            color: {t['text']};
        }}
        .biz-card p {{
            margin: 0;
            font-size: 0.8rem;
            color: {t['text2']};
            line-height: 1.55;
        }}

        /* ── Section Dividers ───────────────────────── */
        .section-title {{
            font-size: 0.72rem;
            font-weight: 700;
            color: {t['text2']};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0.8rem 0 0.3rem 0;
        }}

        /* ── Misc ───────────────────────────────────── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── HTML Component Generators ────────────────────────────────────
def metric_card_html(label: str, value: str, icon: str = "", sub: str = "") -> str:
    """Generate HTML for a KPI metric card."""
    ic = f'<span style="margin-right:3px">{icon}</span>' if icon else ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{ic}{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def recommendation_card_html(
    title: str, genre: str, year: str, score: float, rank: int = 0
) -> str:
    """Generate HTML for a recommendation result card."""
    rank_html = f'<span class="rec-rank">#{rank}</span>' if rank else ""
    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
    return (
        f'<div class="rec-card">'
        f'<div class="rec-title">{rank_html}{title}</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div class="rec-meta">🎭 {genre} &nbsp;·&nbsp; 📅 {year}</div>'
        f'<div class="rec-score">{score_str}</div>'
        f'</div></div>'
    )


def insight_html(text: str) -> str:
    """Generate HTML for an insight callout."""
    return f'<div class="insight-box">💡 {text}</div>'


def business_card_html(title: str, body: str, icon: str = "📊") -> str:
    """Generate HTML for a business-insight card."""
    return (
        f'<div class="biz-card">'
        f'<h4>{icon} {title}</h4>'
        f'<p>{body}</p>'
        f'</div>'
    )
