"""
UI Styling and Custom CSS for ROBOT INTELLIGENCE LAB
Implements a sleek dark mode, glassmorphic metric cards, glowing status pills, and responsive layout.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header Banner */
.main-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #090d16 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #ffffff 0%, #818cf8 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
}

/* Status Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-indigo {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Card Styling */
.metric-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 4px 0;
}

.metric-desc {
    font-size: 0.75rem;
    color: #64748b;
}

/* Alert Boxes */
.alert-box {
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

.alert-info {
    background: rgba(30, 58, 138, 0.25);
    border-left: 4px solid #3b82f6;
    color: #bfdbfe;
}

.alert-warning {
    background: rgba(120, 53, 15, 0.25);
    border-left: 4px solid #f59e0b;
    color: #fde68a;
}

.alert-success {
    background: rgba(6, 78, 59, 0.25);
    border-left: 4px solid #10b981;
    color: #a7f3d0;
}
</style>
"""


def render_header(subtitle: str = "Local AI-Powered Predictive Maintenance and Fault Analytics"):
    """Renders the top application hero banner with badges."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="main-banner">
        <div class="main-title">ROBOT INTELLIGENCE LAB</div>
        <div class="main-subtitle">{subtitle}</div>
        <div class="badge-container">
            <span class="badge badge-green">100% Local Inference • Zero Cloud APIs</span>
            <span class="badge badge-indigo">Kaggle Robot Telemetry Verified</span>
            <span class="badge badge-amber">Leakage-Safe Grouped Splitting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
