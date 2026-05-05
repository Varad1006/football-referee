"""
app.py  —  DeepRef AI Football Referee
Streamlit UI wired to the full RefereePipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import base64
import os
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from pipeline import RefereePipeline, PipelineResult


#  PAGE CONFIG

st.set_page_config(
    page_title="DeepRef · AI Football Referee",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


#  CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #080c10 !important; color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 1100px; padding: 2.5rem 2rem 4rem;
}
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1318; }
::-webkit-scrollbar-thumb { background: #2a3040; border-radius: 3px; }

.hero { text-align: center; padding: 3rem 0 2.5rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: .45rem;
    font-size: .75rem; font-weight: 500; letter-spacing: .12em;
    text-transform: uppercase; color: #f97316;
    background: rgba(249,115,22,.1); border: 1px solid rgba(249,115,22,.25);
    border-radius: 999px; padding: .3rem .9rem; margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 800;
    letter-spacing: -.02em; line-height: 1.08;
    background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .9rem;
}
.hero p { font-size: 1.05rem; color: #64748b; max-width: 540px; margin: 0 auto; line-height: 1.65; font-weight: 300; }

[data-testid="stFileUploader"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] > div { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: 1.5px dashed rgba(249,115,22,.35) !important;
    border-radius: 16px !important; color: #94a3b8 !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(249,115,22,.7) !important; background: rgba(249,115,22,.03) !important;
}
[data-testid="stFileUploadDropzone"] svg { color: #f97316 !important; }

[data-testid="stRadio"] label { font-family: 'DM Sans', sans-serif !important; color: #94a3b8 !important; font-size: .9rem !important; }
[data-testid="stRadio"] > div { flex-direction: row !important; gap: .6rem !important; }

div[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important; font-size: .92rem !important;
    font-weight: 700 !important; letter-spacing: .04em !important;
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: .7rem 2.2rem !important; cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(249,115,22,.35) !important;
}
div[data-testid="stButton"] > button:hover { opacity: .9 !important; }

[data-testid="stVideo"] video {
    border-radius: 14px !important; border: 1px solid rgba(255,255,255,.07) !important; width: 100% !important;
}
.divider { border: none; border-top: 1px solid rgba(255,255,255,.05); margin: 2.5rem 0; }

.pipeline-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    font-size: .75rem; font-weight: 500; letter-spacing: .08em;
    text-transform: uppercase; border-radius: 999px; padding: .25rem .8rem; margin-bottom: 1rem;
}
.pipeline-full     { color: #4ade80; background: rgba(34,197,94,.1);  border: 1px solid rgba(34,197,94,.25); }
.pipeline-fallback { color: #facc15; background: rgba(234,179,8,.1);  border: 1px solid rgba(234,179,8,.25); }

/* ── Result cards ── */
.results-section {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;
    margin-top: 2rem; font-family: 'DM Sans', sans-serif;
}
.r-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 18px; padding: 1.8rem 1.8rem 1.4rem;
    box-shadow: 0 2px 16px rgba(0,0,0,.06);
}
.r-card-title {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 800; color: #0f172a;
    margin-bottom: 1.6rem; display: flex; align-items: center; gap: .5rem;
}
.r-decision-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.3rem; }
.r-decision-label { font-size: .9rem; color: #334155; font-weight: 500; }
.r-badge { font-family: 'Syne', sans-serif; font-size: .8rem; font-weight: 700; border-radius: 8px; padding: .28rem .9rem; letter-spacing: .02em; }
.r-badge.badge-green  { background:#dcfce7; color:#16a34a; border:1px solid #bbf7d0; }
.r-badge.badge-yellow { background:#fef9c3; color:#ca8a04; border:1px solid #fef08a; }
.r-badge.badge-red    { background:#fee2e2; color:#dc2626; border:1px solid #fecaca; }
.r-explanation-block { display: flex; gap: .7rem; align-items: flex-start; padding: .8rem 0; }
.r-exp-icon  { font-size: 1.3rem; flex-shrink: 0; }
.r-exp-title { font-family: 'Syne', sans-serif; font-size: .85rem; font-weight: 700; color: #0f172a; margin-bottom: .35rem; }
.r-exp-text  { font-size: .875rem; color: #64748b; line-height: 1.65; font-weight: 300; }
.r-metric-row { margin-bottom: 1.3rem; }
.r-metric-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: .5rem; }
.r-metric-name { font-family: 'Syne', sans-serif; font-size: .82rem; font-weight: 700; color: #0f172a; }
.r-metric-sub  { font-size: .82rem; color: #64748b; font-weight: 400; margin-left: .4rem; }
.r-pct-badge   { font-family: 'Syne', sans-serif; font-size: .78rem; font-weight: 700; border-radius: 6px; padding: .18rem .55rem; }
.r-bar-track { height: 6px; background: #f1f5f9; border-radius: 999px; overflow: hidden; }
.r-bar-fill  { height: 100%; border-radius: 999px; }
.r-disclaimer {
    font-size: .75rem; color: #94a3b8; margin-top: 1.4rem;
    display: flex; align-items: flex-start; gap: .4rem; line-height: 1.5;
    border-top: 1px solid #f1f5f9; padding-top: 1rem;
}

/* Explainability */
.expl-section {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
    padding: 1.6rem 1.8rem; box-shadow: 0 2px 16px rgba(0,0,0,.06); margin-top: 1.5rem;
}
.expl-title {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 800;
    color: #0f172a; margin-bottom: 1.2rem; display: flex; align-items: center; gap: .5rem;
}
.expl-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.expl-frame { border-radius: 10px; overflow: hidden; position: relative; background: #f8fafc; border: 1px solid #e2e8f0; padding-bottom: 0.6rem; }
.expl-frame img { width: 100%; display: block; border-radius: 10px 10px 0 0; }
.expl-score {
    position: absolute; top: 8px; right: 8px;
    font-family: 'Syne', sans-serif; font-size: .72rem; font-weight: 700;
    background: rgba(0,0,0,.65); color: #f97316; border-radius: 5px; padding: .15rem .45rem;
}
@media (max-width: 700px) {
    .results-section { grid-template-columns: 1fr; }
    .expl-grid { grid-template-columns: 1fr; }
}

/* Custom Highlight Video Button */
.highlight-btn {
    display: flex; justify-content: center; margin-top: 2rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)



#  PIPELINE LOADER (cached)

@st.cache_resource
def load_pipeline(model_dir: str = ".") -> Optional[RefereePipeline]:
    try:
        return RefereePipeline.load(model_dir=model_dir)
    except Exception as e:
        st.error(f"Pipeline load failed: {e}")
        return None


#  VIDEO GENERATOR (Alert + Bounding Boxes)
def create_highlight_video(video_path: str, result: PipelineResult) -> str:
    """Generates an MP4 overlaying a foul alert and border at the point of contact."""
    out_path = "./temp/annotated_output.mp4"
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # avc1 codec is best for web playback in Streamlit
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    peak_idx = result.aggregation.peak_frame_idx
    decision = result.aggregation.decision
    verdict = decision.verdict

    # Determine alert color (BGR format for OpenCV)
    if "Red" in verdict:
        color = (0, 0, 255) # Red
        alert_text = f"FOUL DETECTED: {verdict.upper()}"
    elif "Yellow" in verdict:
        color = (0, 255, 255) # Yellow
        alert_text = f"FOUL DETECTED: {verdict.upper()}"
    else:
        color = (0, 255, 0) # Green
        alert_text = "CLEAN TACKLE / NO FOUL"

    # Alert stays active for 2 seconds
    alert_duration_frames = int(fps * 2)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. DRAW THE ALERT OVERLAY during the peak collision window
        if peak_idx <= frame_idx <= (peak_idx + alert_duration_frames):
            # Draw border
            cv2.rectangle(frame, (0, 0), (w, h), color, 15)
            
            # Draw Text with drop shadow/outline for visibility
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, alert_text, (50, 100), font, 1.5, (0, 0, 0), 6)
            cv2.putText(frame, alert_text, (50, 100), font, 1.5, color, 2)

        # 2. DRAW BOUNDING BOXES 
        if hasattr(result, 'tracking_history') and frame_idx in result.tracking_history:
            for player in result.tracking_history[frame_idx]:
                x1, y1, x2, y2 = map(int, player.bbox)
                
                # Highlight the players involved in the peak collision
                is_involved = False
                if result.events:
                    best_event = max(result.events, key=lambda e: e.duration)
                    if player.track_id in (best_event.player_a_id, best_event.player_b_id):
                        is_involved = True
                
                # Orange if involved, else White
                box_color = (0, 165, 255) if is_involved else (255, 255, 255) 
                thickness = 3 if is_involved else 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                cv2.putText(frame, f"ID: {player.track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return out_path



#  RENDER HELPERS

def _bar_attrs(pct: float):
    if pct >= 80:
        return "#ef4444", "rgba(239,68,68,.12)", "#ef4444"
    elif pct >= 60:
        return "#f97316", "rgba(249,115,22,.12)", "#f97316"
    return "#eab308", "rgba(234,179,8,.12)", "#ca8a04"


def render_results(result: PipelineResult, video_path: str) -> None:
    agg      = result.aggregation
    decision = agg.decision

    bars_html = ""
    for label, sublabel, pct in agg.metrics_as_tuples():
        bar_color, badge_bg, badge_tc = _bar_attrs(pct)
        bars_html += f"""<div class="r-metric-row">
  <div class="r-metric-head">
    <div>
      <span class="r-metric-name">{label}</span>
      <span class="r-metric-sub">{sublabel}</span>
    </div>
    <span class="r-pct-badge"
      style="background:{badge_bg};color:{badge_tc};border:1px solid {badge_tc}44;">
      {pct:.1f}%
    </span>
  </div>
  <div class="r-bar-track">
    <div class="r-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
  </div>
</div>"""

    st.markdown(f"""
<div class="results-section">
  <div class="r-card">
    <div class="r-card-title">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0f172a"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="8" r="4"/>
        <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
        <path d="M9 11l-2 9 5-3 5 3-2-9"/>
      </svg>
      Referee Decision
    </div>
    <div class="r-decision-row">
      <span class="r-decision-label">Decision:</span>
      <span class="r-badge {decision.badge_class}">{decision.verdict}</span>
    </div>
    <div class="r-explanation-block">
      <span class="r-exp-icon">{decision.icon}</span>
      <div>
        <div class="r-exp-title">Explanation:</div>
        <div class="r-exp-text">{decision.explanation}</div>
      </div>
    </div>
    <div class="r-disclaimer">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      AI referee decisions are generated based on similar incidents and official guidelines.
    </div>
  </div>

  <div class="r-card">
    <div class="r-card-title">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#0f172a"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6"  y1="20" x2="6"  y2="14"/>
      </svg>
      Model Analysis
    </div>
    {bars_html}
    <div class="r-disclaimer">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      Percentages are real model outputs — no heuristics.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Explainability strip (Contextual Video Frames)
    peak_idx = agg.peak_frame_idx
    cap = cv2.VideoCapture(video_path)
    
    # Calculate timestamps based on video framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:  
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    idx_before = max(0, peak_idx - int(fps * 2))     # 2 seconds before
    idx_peak   = peak_idx                            # Point of contact
    idx_after  = peak_idx + int(fps * 1.5)           # 1.5 seconds after (adjust if needed)

    if total_frames > 0:
        idx_after = min(idx_after, total_frames - 1)

    frames_to_fetch = [
        ("2s Before Foul", idx_before),
        ("Point of Contact", idx_peak),
        ("After Foul", idx_after)
    ]

    frames_html = ""
    for label, f_idx in frames_to_fetch:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(f_idx))
        ret, frame = cap.read()
        if ret:
            # Resize for the UI so it doesn't pass huge base64 strings to frontend
            h, w = frame.shape[:2]
            if w > 600:
                scale = 600 / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64    = base64.b64encode(buf.tobytes()).decode()
            
            frames_html += f"""
            <div class="expl-frame">
              <img src="data:image/jpeg;base64,{b64}" alt="{label} frame {f_idx}"/>
              <div style="text-align: center; margin-top: 0.6rem; font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #0f172a;">{label}</div>
              <span class="expl-score">Frame {f_idx}</span>
            </div>"""

    cap.release()

    if frames_html:
        st.markdown(f"""
<div class="expl-section">
  <div class="expl-title">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0f172a"
         stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="23 7 16 12 23 17 23 7"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
    </svg>
    Key Contextual Frames
  </div>
  <div class="expl-grid">{frames_html}</div>
</div>""", unsafe_allow_html=True)


def save_upload(uploaded_file) -> str:
    os.makedirs("./temp", exist_ok=True)
    path = "./temp/clip.mp4"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path



#  HERO

st.markdown("""
<div class="hero">
    <div class="hero-badge">⚽ AI Powered</div>
    <h1>AI Football Referee</h1>
    <p>Real-time YOLO tracking + temporal ensemble classification.
       Every metric is driven by your models — no fake numbers.</p>
</div>
""", unsafe_allow_html=True)



#  FILE UPLOADER & SESSION STATE


# 1. Initialize Streamlit Memory (Session State)
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
    st.session_state.video_path = None
    st.session_state.current_file_name = None

uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "avi", "mov", "3gp"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0 2rem;color:#334155;font-size:.85rem;">'
        'Supported formats: <strong style="color:#475569">MP4 · AVI · MOV · 3GP</strong>'
        '</div>',
        unsafe_allow_html=True,
    )


#  ANALYSIS

if uploaded_file is not None:

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.video(uploaded_file)

    # 2. Reset memory if a completely new video is uploaded
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.analysis_result = None
        st.session_state.video_path = None
        st.session_state.current_file_name = uploaded_file.name

    col_btn, _ = st.columns([2, 8])
    with col_btn:
        analyze = st.button("⚡  Analyse Clip")

    # 3. Handle the "Analyse" button click
    if analyze:
        video_path = save_upload(uploaded_file)

        with st.spinner("Loading models…"):
            pipeline = load_pipeline(model_dir=".")

        if pipeline is None:
            st.error("Failed to initialise the pipeline. Check model files.")
            st.stop()

        with st.spinner("Running YOLO detection → tracking → ensemble classification…"):
            result = pipeline.analyse(video_path)
            
            # SAVE the result and video path to memory so it survives button clicks!
            st.session_state.analysis_result = result
            st.session_state.video_path = video_path

    # 4. Render the UI ONLY if we have a result in memory
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        video_path = st.session_state.video_path

        # Pipeline mode badge 
        mode_cls = "pipeline-full" if result.pipeline_mode == "full" else "pipeline-fallback"
        mode_lbl = "⚡ Full pipeline · YOLO + Tracking" if result.pipeline_mode == "full" \
                   else "⚠ Fallback · Direct classification"
        st.markdown(f'<span class="pipeline-badge {mode_cls}">{mode_lbl}</span>',
                    unsafe_allow_html=True)

        if result.warning:
            st.warning(result.warning)

        if result.events:
            best  = max(result.events, key=lambda e: e.duration)
            st.markdown(
                f'<p style="color:#64748b;font-size:.85rem;margin-bottom:.5rem;">'
                f'Detected <strong style="color:#f97316">{len(result.events)}</strong> interaction(s). '
                f'Peak event at frame <strong style="color:#f97316">{best.peak_frame}</strong> '
                f'({best.duration} frames).</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        render_results(result, video_path)
        
        # Annotated Output Video Button 
        st.markdown('<div class="highlight-btn">', unsafe_allow_html=True)
        
        # Because the results are in session_state, clicking this button won't wipe the screen
        if st.button("🎬 Generate Annotated Highlight Video", type="secondary"):
            with st.spinner("Rendering video with foul alerts and bounding boxes..."):
                output_video_path = create_highlight_video(video_path, result)
                if os.path.exists(output_video_path):
                    st.success("Highlight video generated successfully!")
                    st.video(output_video_path)
                else:
                    st.error("Failed to generate video.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Debug footer 
        agg = result.aggregation
        st.markdown(
            f'<p style="color:#334155;font-size:.75rem;margin-top:1.2rem;">'
            f'foul_prob={agg.foul_prob:.3f} · severity={agg.severity:.3f} · '
            f'consistency={agg.consistency:.3f} · elapsed={result.elapsed_seconds:.1f}s</p>',
            unsafe_allow_html=True,
        )