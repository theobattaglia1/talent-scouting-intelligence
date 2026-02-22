from __future__ import annotations


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #f6f4ef;
  --surface: #ffffff;
  --surface-2: #f2efe7;
  --text: #151a17;
  --muted: #5a615d;
  --border: #d7d7cf;
  --accent: #145a42;
  --accent-soft: #e2f2ec;
  --danger: #b3463e;
  --warning: #995f00;
  --success: #1f7a3e;
  --radius: 14px;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1413;
    --surface: #161d1b;
    --surface-2: #1f2926;
    --text: #edf3ef;
    --muted: #b5c3bd;
    --border: #2c3833;
    --accent: #5ad0a0;
    --accent-soft: #163529;
    --danger: #ef8d86;
    --warning: #f7bf68;
    --success: #66d88d;
  }
}

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 10% 0%, rgba(20,90,66,0.08), transparent 35%),
    radial-gradient(circle at 95% 90%, rgba(130,160,40,0.06), transparent 40%),
    var(--bg);
  color: var(--text);
}

h1, h2, h3 {
  font-family: 'Fraunces', serif;
  letter-spacing: -0.015em;
}

.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  border-right: 1px solid var(--border);
}

.metric-card, .panel, .state-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  padding: var(--space-4) var(--space-5);
  box-shadow: 0 10px 24px rgba(0,0,0,0.06);
  animation: fadeIn 240ms ease-in;
}

.metric-label {
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: var(--space-1);
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 800;
  line-height: 1.1;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  border-radius: 999px;
  padding: 0.2rem 0.65rem;
  border: 1px solid var(--border);
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  text-transform: uppercase;
}

.chip-breaking {
  background: rgba(31, 122, 62, 0.16);
  color: var(--success);
}

.chip-emerging {
  background: rgba(153, 95, 0, 0.15);
  color: var(--warning);
}

.chip-early {
  background: rgba(20, 90, 66, 0.12);
  color: var(--accent);
}

.chip-baseline {
  background: rgba(90, 97, 93, 0.16);
  color: var(--muted);
}

.chip-flag {
  background: rgba(179, 70, 62, 0.14);
  color: var(--danger);
}

.empty-state {
  text-align: center;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  padding: 2.4rem 1.2rem;
  background: var(--surface-2);
}

.help-copy {
  color: var(--muted);
  font-size: 0.92rem;
}

.small-note {
  color: var(--muted);
  font-size: 0.82rem;
}

[data-testid="stToolbar"] {
  right: 1rem;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid var(--border);
  padding: 0.48rem 0.9rem;
  font-weight: 600;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(125deg, #145a42, #1b7e5d);
  border-color: #145a42;
  color: #f6fffb;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""


def stage_chip(stage: str) -> str:
    stage_clean = str(stage).strip().lower()
    cls = "chip-early"
    if stage_clean == "breaking":
        cls = "chip-breaking"
    elif stage_clean == "emerging":
        cls = "chip-emerging"
    elif stage_clean == "baseline":
        cls = "chip-baseline"
    return f"<span class='chip {cls}'>{stage_clean or 'unknown'}</span>"
