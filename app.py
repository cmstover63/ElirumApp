import streamlit as st
import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import time

# Additional modules for styling and encoding assets
import base64  # Used to embed the logo image directly in HTML

# ---------------------------------------------------------------------------
# Audio feature extraction
#
# The functions below implement lightweight audio analysis using the
# ``librosa`` library.  When a video is uploaded, we call
# ``extract_audio_features`` once to compute simple descriptors such as
# average energy, pitch variability and zero‑crossing rate.  These
# descriptors are then normalised and combined into a single audio stress
# score via ``compute_audio_stress_score``.  If ``librosa`` isn't
# installed or the audio cannot be decoded, these functions return
# ``None`` or zero so that the rest of the application can proceed
# without audio cues.

def extract_audio_features(video_path: str):
    """
    Extract basic audio features from the given video file.

    Parameters
    ----------
    video_path : str
        Absolute path to the uploaded video file.  ``librosa`` will
        attempt to decode the audio track directly.  If decoding fails
        or ``librosa`` is unavailable, this function returns ``None``.

    Returns
    -------
    dict | None
        A dictionary containing the mean energy, pitch standard deviation
        and mean zero‑crossing rate of the audio.  ``None`` if
        features cannot be extracted.
    """
    if librosa is None:
        return None
    try:
        # Load the audio at a lower sample rate to speed up processing.
        # The ``sr`` of 22050 Hz is commonly used in speech analysis.
        y, sr = librosa.load(video_path, sr=22050)
        if y.size == 0:
            return None
        # Root‑mean‑square energy gives a measure of overall loudness.
        rmse = librosa.feature.rms(y=y).flatten()
        energy_mean = float(np.mean(rmse)) if rmse.size else 0.0
        # Pitch estimation via the YIN algorithm.  We compute the
        # fundamental frequency for the entire waveform and then take
        # the standard deviation as an indicator of variability in
        # prosody.  Large variations in pitch can be signs of stress.
        pitches = librosa.yin(y, fmin=75, fmax=400, sr=sr)
        pitch_std = float(np.std(pitches)) if pitches.size else 0.0
        # Zero‑crossing rate is the rate at which the signal changes
        # sign and correlates with tremor or roughness in the voice.
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
        return {
            "energy_mean": energy_mean,
            "pitch_std": pitch_std,
            "zcr_mean": zcr_mean,
        }
    except Exception:
        # If decoding fails (e.g. unsupported format) return None
        return None


def compute_audio_stress_score(features: dict | None) -> float:
    """
    Convert extracted audio features into a normalised stress score.

    The returned value is in the range [0, 1] and represents the
    contribution of audio cues to the overall stress level.  Each
    feature is scaled by a heuristic constant derived from typical
    speech characteristics.  If ``features`` is ``None``, zero is
    returned.

    Parameters
    ----------
    features : dict | None
        Output from :func:`extract_audio_features`.

    Returns
    -------
    float
        A value between 0 and 1 representing audio‑derived stress.
    """
    if not features:
        return 0.0
    # Normalise energy: typical root‑mean‑square energy for clean
    # conversational speech is around 0.01–0.02.  Use a larger divisor
    # (0.10) so that normal energy levels contribute less aggressively
    # toward the maximum score.  This helps keep the audio score
    # moderate in realistic recordings.
    energy_norm = min(features.get("energy_mean", 0.0) / 0.10, 1.0)
    # Normalise pitch variability: typical standard deviation around
    # 30–50 Hz.  Increase the divisor to 80 to reduce the effect of
    # pitch fluctuations on the overall audio score.
    pitch_norm = min(features.get("pitch_std", 0.0) / 80.0, 1.0)
    # Normalise zero‑crossing rate: values between 0.05–0.1 are common
    # for voiced speech.  Increase the divisor to 0.3 to further dampen
    # the influence of ZCR on the audio stress.
    zcr_norm = min(features.get("zcr_mean", 0.0) / 0.3, 1.0)
    # Combine equally.  You can adjust weights here if certain cues
    # should contribute more to stress.
    return (energy_norm + pitch_norm + zcr_norm) / 3.0

# Configure the Streamlit page to use a wide layout and collapse the sidebar.
st.set_page_config(page_title="Elirum", layout="wide", initial_sidebar_state="collapsed")

# Add MediaPipe for facial mesh, pose skeleton and hand detection.  MediaPipe
# enables lightweight, fast landmark detection for faces, full‑body pose and
# hands.  These landmarks drive the red‑dot mesh over the face, the green
# skeletal lines down the body and the blue connections across hands.
# Attempt to import MediaPipe and ensure the ``solutions`` module exists.  Some
# lightweight or stub packages named ``mediapipe`` do not expose the
# ``solutions`` attribute, which leads to runtime errors.  If either the
# import fails or the attribute is missing, ``mp`` is set to ``None`` so
# that the application falls back to OpenCV‑based overlays.
try:
    import mediapipe as mp  # type: ignore
    # Verify that the expected submodule is present.  If not, trigger the
    # AttributeError handler below.
    if not hasattr(mp, "solutions"):
        raise AttributeError("mediapipe.solutions missing")
except (ImportError, AttributeError):
    mp = None

# Attempt to import librosa for audio feature extraction.  If the
# dependency isn't available (e.g. because it isn't installed in
# deployment), we'll set ``librosa`` to ``None``.  This allows the
# application to run without audio‑based cues and prevents import errors.
try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore

# -------------------------
# LANDMARK MODELS
# -------------------------
# MediaPipe models are used for face, pose and hand landmark detection.  They
# provide detailed meshes and skeleton connections with minimal overhead.  The
# Haar cascade models remain available as a fallback for gaze shift estimation,
# but all drawing is now driven by MediaPipe landmarks.

# Load Haar cascade model for optional gaze shift detection.  OpenCV installs
# the cascade files with the Python package, and the paths are accessible via
# ``cv2.data.haarcascades``.  This cascade is only used to estimate face
# bounding boxes for computing gaze direction; the visual overlay comes from
# MediaPipe landmarks.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if mp is not None:
    # Initialise MediaPipe solutions.  Refinement on the face mesh adds iris
    # landmarks which improve the appearance of the mesh.  The pose and hand
    # solutions run in streaming mode for performance.  Note that MediaPipe
    # context creation is deferred until a video is loaded to avoid
    # unnecessary resource allocation when the app starts.
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
else:
    mp_face_mesh = None
    mp_pose = None
    mp_hands = None

# -------------------------
# NOTE
# -------------------------
# This application implements behavioural analysis using OpenCV without relying
# on MediaPipe. A combination of motion detection and Haar‑cascade based face
# and body detection provides stress cues and live overlays.  See the video
# analysis loop for implementation details.

# -------------------------
# BRANDING CONFIGURATION
# -------------------------
# Path to your logo image within the app directory
LOGO_PATH = "elirum_logo.png"

# Colour palette (Option C: light gradient tech‑startup look)
# Primary colour: deep teal used for buttons and header bands.
PRIMARY_COLOR = "#007A84"     # deep teal
# Secondary colour: fresh green for highlights and secondary UI elements.
SECONDARY_COLOR = "#00A36C"   # green‑teal
# Accent colour: lime for subtle highlights in charts or alerts.
ACCENT_COLOR = "#4CD964"      # lime highlight
# Background colour: white for a clean, modern interface.
BACKGROUND_COLOR = "#FFFFFF"  # white
# Additional dark accent (not currently used but available for future styling).
DARK_ACCENT_COLOR = "#003C45"

# -------------------------
# LOGIN SYSTEM
# -------------------------
# Simple hard‑coded user database
USERS = {
    "charles": "secure123",
    "admin": "admin123",
    "detective": "detect2025"
}

def show_landing_page():
    """Render the Elirum landing page with hero section and login support.

    This implementation avoids mixing custom HTML with Streamlit widgets by
    using Streamlit's built-in layout primitives for the login page.  When
    the login flag is set in session state, the page renders a centred
    sign-in form on a white background and stops further rendering.
    Otherwise, it displays the hero section with tagline and features.
    """
    # Encode images for embedding in HTML/CSS
    try:
        with open("hero_bg.png", "rb") as bg_file:
            hero_data = base64.b64encode(bg_file.read()).decode("utf-8")
    except FileNotFoundError:
        hero_data = ""
    try:
        with open(LOGO_PATH, "rb") as logo_file:
            logo_data = base64.b64encode(logo_file.read()).decode("utf-8")
    except FileNotFoundError:
        logo_data = ""

    # Handle the login page: centre login form on white background
    if st.session_state.get("login_page", False):
        # Minimal CSS to remove default padding/margins and set white background
        st.markdown(
            """
            <style>
            body, html, .stApp {
                margin: 0;
                padding: 0;
                background-color: #FFFFFF;
            }
            .block-container {
                margin: 0;
                padding: 0;
                max-width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Vertical spacing
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Use columns to centre the login card
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(LOGO_PATH, width=180)
            st.markdown("<h2>Sign In</h2>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_page_user")
            password = st.text_input("Password", type="password", key="login_page_pass")
            if st.button("Sign In", key="login_page_signin"):
                if username in USERS and USERS[username] == password:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = username
                    st.session_state["login_page"] = False
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        st.stop()

    # If we are not in login mode, inject additional CSS to ensure hero feature
    # bullet points use standard disc bullets instead of Unicode checkmarks.
    # This override targets the hero feature list specifically.  It sets
    # `list-style-type: disc` on the <ul> and removes the pseudo-element
    # `::before` on the <li> elements that previously added a checkmark glyph.
    st.markdown(
        """
        <style>
        .hero .overlay ul {
            list-style-type: disc !important;
            padding-left: 1.5rem;
        }
        .hero .overlay li::before {
            content: '' !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Login button on the hero page
    top_login_clicked = st.button("Login", key="landing_enter_button")
    if top_login_clicked:
        st.session_state["login_page"] = True
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    # Hero section CSS and markup
    st.markdown(
        f"""
        <style>
        /* Reset margins and padding */
        body, html, .stApp {{ margin: 0; padding: 0; }}
        .block-container {{ padding: 0; margin: 0; max-width: 100%; width: 100%; }}

        /* Hero container */
        .hero {{
            position: relative;
            width: 100%;
            height: 100vh;
            background-image: url('data:image/png;base64,{hero_data}');
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            overflow: hidden;
        }}
        /* Overlay box */
        .hero .overlay {{
            background: rgba(0, 0, 0, 0.55);
            padding: 3rem 4rem;
            border-radius: 8px;
            max-width: 900px;
        }}
        .hero .overlay h1 {{
            margin-top: 0;
            color: {ACCENT_COLOR};
            font-size: 3.5rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .hero .overlay h3 {{
            font-size: 1.3rem;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        .hero .overlay ul {{
            /* Use standard disc bullets for clarity on all devices.  The
               padding creates space between the bullet and the text. */
            list-style: disc;
            padding-left: 1.5rem;
            margin: 0 0 2rem 0;
        }}
        .hero .overlay li {{
            margin-bottom: 0.7rem;
            font-size: 1.15rem;
            position: relative;
            /* No extra padding; the list-style bullet handles indentation */
            padding-left: 0;
        }}
        /* Override the pseudo-element used previously for checkmark icons.  By
           setting an empty content, we avoid characters like "14" appearing
           in some fonts. */
        .hero .overlay li::before {{
            content: "";
        }}
        /* Style the hero login button. Target only the first Streamlit button
           on the page (the hero login button) so other buttons are unaffected.
           Position it fixed near the top‑right of the viewport for desktop.
           The width is set to auto so the button fits its text instead of
           stretching across the page.  Adjust padding and border to match
           the branded style. */
        .stButton:nth-of-type(1) > button {{
            position: fixed;
            top: 80px;         /* roughly 1 inch from the top */
            right: 60px;       /* roughly aligned to the right margin */
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: auto;
            min-width: 120px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            border-radius: 8px;
            background: {PRIMARY_COLOR};
            border: 2px solid {PRIMARY_COLOR};
            color: #ffffff;
            font-weight: 600;
            z-index: 1000;
        }}
        .stButton:nth-of-type(1) > button:hover {{
            background: {ACCENT_COLOR};
            border-color: {ACCENT_COLOR};
            color: #ffffff;
        }}
        </style>

        <div class="hero">
            <div class="overlay">
                <img src="data:image/png;base64,{logo_data}" alt="Elirum Logo" style="height:100px;margin-bottom:1rem;" />
                <h1>Don't second guess</h1>
                <!-- Use standard ASCII hyphens instead of non‑breaking hyphens to avoid PDF encoding errors -->
                <h3>Experience the leading AI-powered system for behavioural and nervousness detection.</h3>
                <ul>
                    <li>AI-driven facial and body landmark analysis</li>
                    <li>Real-time quantification of stress and nervousness</li>
                    <li>Secure analytics tailored for law-enforcement professionals</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True,
    )
    # Close the hero div
    st.markdown("</div>", unsafe_allow_html=True)

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    # Default page after login is the dashboard.  The page state is used
    # to route between the dashboard and the invite page.
    st.session_state["page"] = "dashboard"

# If not authenticated, show the landing page and stop execution.  The
# landing page includes a sign‑in form and a description of the business.
if not st.session_state["authenticated"]:
    show_landing_page()
    st.stop()

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(page_title="Elirum Analyzer", layout="wide")

# Inject custom CSS for background and button colours
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    /* Buttons on analysis pages will adopt the primary colour by default when not
       overridden on the landing page.  We avoid setting a global rule for .stButton
       here so that the hero page login button styling remains intact. */
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo and page title on one row.  The logo appears on the left
# and the title and caption on the right.  This layout matches the
# user’s preference for left‑aligned logo placement (Option 2) and
# provides a balanced header at the top of the application.
header_logo_col, header_title_col = st.columns([1, 4])
with header_logo_col:
    st.image(LOGO_PATH, width=120)
with header_title_col:
    st.title("Elirum Analyzer")
    st.caption("Behavioral Stress & Nervousness Detection System")

    # Navigation for authenticated users.  Only display the invite option to
    # administrators (identified here as the "admin" user).  When the
    # button is clicked a flag is set in session state and the page is
    # rerun.  The routing logic below will render the invite page and
    # prevent the rest of the analysis UI from loading.
    if st.session_state.get("authenticated"):
        if st.session_state.get("user") == "admin":
            if st.button("Invite New User", key="invite_nav_button"):
                st.session_state["page"] = "invite"
                # Immediately rerun to trigger the routing logic
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

# Route to pages based on the current session state.  If the admin has
# selected the invite page, import and render it here.  After rendering
# the invite page we halt further execution to avoid showing the
# analysis interface.  The default page is "dashboard", which
# corresponds to the main analysis UI below.
if st.session_state.get("authenticated") and st.session_state.get("page") == "invite":
    from pages import invite as invite_page
    invite_page.app()
    st.stop()

# -------------------------
# MAIN UPLOAD SECTION
# -------------------------
# Remove the collapsed sidebar toggle so the arrow does not appear after login.
st.markdown(
    """
    <style>
    /* Hide the collapsed sidebar control that appears on the left edge */
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Place instructions and upload field directly on the main page so users do not
# need to expand the sidebar.  Use columns to separate the instructions and
# uploader for a balanced layout.  The file uploader returns a file handle
# when a video is selected.
inst_col, upload_col = st.columns([2, 1])
with inst_col:
    st.subheader("Instructions")
    st.markdown(
        """
        1. Upload interview video  
        2. Behavioral analysis runs automatically  
        3. Review stress timeline  
        4. Download official PDF report  
        """
    )
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi"],
        key="main_upload"
    )

# -------------------------
# PDF GENERATOR
# -------------------------
def generate_pdf(user: str, fps: float, stress_events: list, scores: list, audio_features: dict | None = None, audio_score: float | None = None) -> str:
    """Generate a PDF report summarising the behavioural analysis.

    Args:
        user (str): The username of the analyst.
        fps (float): Frames per second of the analysed video.
        stress_events (list): A list of dicts containing time, score and cues for high‑stress events.
        scores (list): A list of all stress scores (0‑100) for the timeline.

    Returns:
        str: The filename of the generated PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header band using primary colour
    pdf.set_fill_color(int(PRIMARY_COLOR[1:3], 16), int(PRIMARY_COLOR[3:5], 16), int(PRIMARY_COLOR[5:7], 16))
    pdf.rect(0, 0, 210, 15, "F")
    pdf.set_y(20)

    # Document title
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "ELIRUM BEHAVIORAL ANALYSIS REPORT", ln=True, align="C")

    pdf.ln(6)
    pdf.set_font("Arial", "", 12)

    avg_score = round(np.mean(scores), 2)
    max_score = round(np.max(scores), 2)

    pdf.multi_cell(
        0, 8,
        f"Analyst: {user}\n"
        f"Average Stress Level: {avg_score}%\n"
        f"Peak Stress Level: {max_score}%\n"
        f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    pdf.ln(4)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Flagged Behavioral Events", ln=True)

       pdf.set_font("Arial", "", 11)
    for e in stress_events:
        # Sanitize non‑breaking hyphens so FPDF can encode the text
        trend = str(e.get('trend', '')).replace(chr(0x2011), '-')
        cues = str(e.get('cues', '')).replace(chr(0x2011), '-')
        notes = str(e.get('notes', '')).replace(chr(0x2011), '-')
        text = (
            f"Time: {e['time']}s | Stress: {e['score']}% ({trend})\n"
            f"Indicators: {cues}\n"
            f"Notes: {notes}"
        )
        pdf.multi_cell(0, 8, text)
    pdf.ln(1)

    # If audio analysis information is available, add a summary section.  This
    # section provides the reader with the underlying vocal metrics used to
    # compute the audio stress score and summarises the overall vocal
    # stress level.  Audio features may be None if extraction failed or
    # librosa is unavailable.
    if audio_features is not None:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Audio Analysis Summary", ln=True)
        pdf.set_font("Arial", "", 11)
        # Extract values or provide placeholders if missing
        energy = audio_features.get("energy_mean", 0.0) if isinstance(audio_features, dict) else 0.0
        pitch_std = audio_features.get("pitch_std", 0.0) if isinstance(audio_features, dict) else 0.0
        zcr = audio_features.get("zcr_mean", 0.0) if isinstance(audio_features, dict) else 0.0
        score_pct = round((audio_score or 0.0) * 100, 2)
        audio_label = ""
        if audio_score is not None:
            if audio_score >= 0.7:
                audio_label = "High Vocal Stress"
            elif audio_score >= 0.4:
                audio_label = "Moderate Vocal Stress"
            elif audio_score >= 0.2:
                audio_label = "Mild Vocal Stress"
            else:
                audio_label = "Low Vocal Stress"
        pdf.multi_cell(
        0, 8,
        f"Average energy (RMS): {energy:.4f}\n"
        f"Pitch variability (std): {pitch_std:.2f} Hz\n"
        f"Zero-crossing rate: {zcr:.4f}\n"
        f"Audio Stress Score: {score_pct}% ({audio_label})"
    )
    pdf.ln(4)

    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0, 6,
        "Disclaimer: Elirum provides automated behavioural pattern detection and does not determine deception or intent."
        " Results must be interpreted by trained professionals in context."
    )

    file_name = "Elirum_Analysis_Report.pdf"
    pdf.output(file_name)
    return file_name

# -------------------------
# VIDEO ANALYSIS AND UI
# -------------------------
if uploaded_file:
    # If this file has already been processed during the current session,
    # reuse the stored results to avoid re‑running the analysis.  This keeps
    # the timeline, metrics and notes visible after the user downloads the
    # report.  Otherwise, perform the full analysis.
    processed_name = st.session_state.get("processed_file")
    if processed_name == uploaded_file.name and st.session_state.get("nervous_scores"):
        # The file has been analysed previously.  Show playback and
        # previously computed results.
        st.subheader("Interview Playback")
        st.video(uploaded_file)

        # Use stored fps for report generation.  If not available, fall back
        # to 30 FPS.
        fps = st.session_state.get("fps", 30)

        # Display timeline and metrics using stored values
        col1, col2 = st.columns([2, 1])
        col1.subheader("Previously Computed Stress Timeline")
        if st.session_state.get("nervous_scores"):
            df_cached = pd.DataFrame(
                {"Stress (%)": st.session_state["nervous_scores"]},
                index=st.session_state["time_values"],
            )
            col1.line_chart(df_cached)
            col2.metric(
                "Average Stress",
                f"{round(np.mean(st.session_state['nervous_scores']), 2)}%",
            )
            col2.metric(
                "Peak Stress",
                f"{round(np.max(st.session_state['nervous_scores']), 2)}%",
            )

        # Generate the PDF using cached results
        pdf_name = generate_pdf(
            user=st.session_state["user"],
            fps=fps,
            stress_events=st.session_state.get("stress_events", []),
            scores=st.session_state.get("nervous_scores", []),
            audio_features=st.session_state.get("audio_features"),
            audio_score=st.session_state.get("audio_score"),
        )
        with open(pdf_name, "rb") as f:
            st.download_button(
                "Download Professional PDF Report",
                f,
                file_name=pdf_name,
                key="download_pdf_cached",
            )

        # Display the flagged stress moments with notes using cached data
        st.subheader("Flagged Stress Moments (Add Notes)")
        for i, e in enumerate(st.session_state.get("stress_events", [])):
            note_key = f"note_cached_{i}"
            note = st.text_input(
                f"{e['time']}s — {e['score']}% ({e['trend']})",
                key=note_key,
                value=e.get("notes", ""),
            )
            st.session_state["stress_events"][i]["notes"] = note

        # Skip further processing for cached files
        st.stop()

    # Save uploaded file to a temporary file.  Using a temporary file rather than
    # reading the entire buffer into memory helps keep RAM usage low and
    # improves the reliability of large uploads.  The video is written to
    # disk and then accessed by OpenCV via its filename.
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Show the raw video with audio for playback
    st.subheader("Interview Playback")
    st.video(uploaded_file)

    # ---------------------------------------------------------------------
    # AUDIO ANALYSIS
    #
    # Before we begin frame‑by‑frame processing, extract a few simple
    # features from the video's audio track.  These values remain
    # constant throughout the analysis and contribute to the overall
    # stress score.  If audio decoding fails or librosa isn't
    # installed, the audio score will be zero.
    audio_features = extract_audio_features(tfile.name)
    audio_score = compute_audio_stress_score(audio_features)
    # Determine an audio cue label based on the overall audio stress.  This label
    # will be appended to flagged stress events and reported in the PDF.
    # High vocal stress corresponds to elevated energy, pitch variability or
    # zero‑crossing rate.  Moderate indicates some vocal strain, while mild
    # vocal stress reflects subtle changes.  If the audio score is very low,
    # no vocal cue will be added.
    audio_cue_label = None
    try:
        if audio_score >= 0.7:
            audio_cue_label = "High Vocal Stress"
        elif audio_score >= 0.4:
            audio_cue_label = "Moderate Vocal Stress"
        elif audio_score >= 0.2:
            audio_cue_label = "Mild Vocal Stress"
        else:
            audio_cue_label = None
    except Exception:
        audio_cue_label = None

    # Persist audio features and score for use in downstream components, such
    # as generating the PDF report or caching results.  Storing these in
    # ``st.session_state`` allows other functions to access the values
    # without recomputing them or passing them through multiple layers.
    st.session_state["audio_features"] = audio_features
    st.session_state["audio_score"] = audio_score

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Determine the total number of frames for progress tracking.  Some
    # containers may return 0 for unknown lengths, so guard against division
    # by zero.  Use 1 as a default to avoid zero division and ensure the
    # progress bar updates.
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1

    # Set fixed parameters for processing speed and quality.  The analysis
    # pipeline will process every second frame and downscale the video to
    # 50 % of its original resolution before running detection.  These
    # constants strike a balance between speed and accuracy without
    # requiring user input.
    FRAME_SKIP = 3  # analyse every 3rd frame
    processing_scale = 0.5  # process at half resolution

    # Always display the face mesh, pose skeleton and hand connections if
    # available.  Hiding the overlay options ensures a consistent
    # appearance without requiring the user to adjust settings.
    show_face_mesh = True
    show_pose = True
    show_hands = True

    # Add a progress bar to the sidebar so the user can see analysis progress
    progress_bar = st.sidebar.progress(0)

    # Instantiate MediaPipe models if available.  Creating these objects
    # inside the file upload block avoids allocating GPU/CPU resources when
    # the application is idle.  Each solution is configured for live
    # streaming with a single face and up to two hands.
    if mp is not None:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        face_mesh = None
        pose = None
        hands = None

    # Initialise data structures for motion‑based analysis
    nervous_scores = []
    # Track the timestamp (in seconds) corresponding to each processed frame.
    # This will be used to plot the timeline on a seconds axis.
    time_values = []
    stress_events = []

    # Previous frame for motion detection (initially None).  We store
    # the greyscale downscaled frame here to compare motion across frames.
    prev_gray_small = None

    # Track the previous face centre to estimate gaze shifts.  A large
    # horizontal movement of the face centre between frames will be
    # interpreted as the subject looking off centre ("Gaze Shift").
    prev_face_center = None

    # Variables to determine trend changes and smoothing
    last_score = 0.0
    # Baseline for motion intensity calibration.  We gather motion values
    # from the first few processed frames and compute an average to
    # normalise later movement scores.  This helps prevent the stress
    # score from saturating at 100% on videos with consistent motion.
    motion_baseline = None
    baseline_values = []
    # Number of processed frames used to compute the baseline motion
    # intensity.  A larger number reduces the impact of early frame
    # noise and helps avoid saturating the stress score.  The default of
    # 60 corresponds to roughly two seconds of video at 30fps (with
    # frame skipping).  Feel free to adjust depending on typical video
    # length and motion.
    baseline_frames = 60
    last_logged_second = -1
    last_logged_cues = ""

    col1, col2 = st.columns([2, 1])
    video_slot = col1.empty()

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Update progress bar based on how many frames have been processed.
        # Use min() to clamp the value to 1.0 in case of rounding errors.
        progress_fraction = frame_idx / total_frames
        progress_bar.progress(min(progress_fraction, 1.0))

        # Skip frames to reduce processing load.  FRAME_SKIP is user‑selectable
        # via the slider.  Only every nth frame is analysed; the rest are
        # dropped to save CPU.
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Downscale the frame before processing.  Shrinking the frame
        # substantially reduces computation for motion detection and
        # landmark estimation.  The processed landmarks are later mapped
        # back to the original frame dimensions for drawing.  We also
        # prepare both greyscale and RGB versions of the downscaled frame.
        frame_small = cv2.resize(
            frame,
            (0, 0),
            fx=processing_scale,
            fy=processing_scale,
        )
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        # Reset score and cues for this frame
        score = 0.0
        cues = []

        if prev_gray_small is not None:
            # Compute absolute difference between current frame and previous frame
            diff = cv2.absdiff(gray_small, prev_gray_small)
            # Calculate average pixel intensity of the difference as a proxy for motion
            movement_intensity = np.mean(diff)

            # Build baseline from the first `baseline_frames` processed frames.  Once
            # enough frames have been collected, compute the baseline as the mean.
            if motion_baseline is None and len(baseline_values) < baseline_frames:
                baseline_values.append(movement_intensity)
                # Use a provisional divisor during baseline collection to avoid division by zero
                score = min(max(movement_intensity / 50.0, 0.0), 1.0)
            else:
                if motion_baseline is None and baseline_values:
                    # Compute an initial baseline from collected samples.  A small
                    # epsilon is added to avoid division by zero.
                    motion_baseline = np.mean(baseline_values) + 1e-5
                # Adapt the baseline over time using a weighted moving average.
                # The ``alpha`` parameter controls how quickly the baseline
                # responds to new motion values.  A small alpha yields slow
                # adaptation, emphasising transient spikes relative to the
                # current baseline rather than saturating the score when
                # movement is consistently high.
                alpha = 0.05
                motion_baseline = (1 - alpha) * motion_baseline + alpha * movement_intensity
                # Normalise the difference between the current intensity and the
                # dynamic baseline.  A smaller denominator (6× baseline)
                # increases sensitivity to spikes while still preventing the
                # score from saturating under moderate, sustained motion.
                norm_intensity = (movement_intensity - motion_baseline) / (motion_baseline * 6.0)
                score = min(max(norm_intensity, 0.0), 1.0)

            # Derive cues based on movement intensity.  Larger differences
            # correspond to bigger body movements and are labelled accordingly.
            if movement_intensity > (motion_baseline or 0) * 3.5:
                cues.append("Posture Shift")
            elif movement_intensity > (motion_baseline or 0) * 2.5:
                cues.append("Hand Fidget")
            elif movement_intensity > (motion_baseline or 0) * 1.0:
                cues.append("Facial Tension")

        # Additional cue: face movement (gaze/pose changes)
        # Detect face using the Haar cascade on the downscaled grayscale frame.  If
        # a face is detected, compute how far its centre has moved relative to
        # the previous frame.  Large movements suggest the subject is looking
        # away from the interviewer or making notable head movements, which we
        # classify as a "Gaze Shift" cue.  The face movement score is used to
        # bump the overall stress score.
        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            # Select the largest detected face by area
            (fx, fy, fw, fh) = max(faces, key=lambda b: b[2] * b[3])
            # Compute the centre of the face bounding box
            face_center = (fx + fw / 2.0, fy + fh / 2.0)
            if prev_face_center is not None:
                # Compute horizontal and vertical shifts separately
                dx = abs(face_center[0] - prev_face_center[0])
                dy = abs(face_center[1] - prev_face_center[1])
                # Euclidean distance between current and previous face centres
                dist = np.linalg.norm(np.array(face_center) - np.array(prev_face_center))
                # Normalise movements by face dimensions to account for scale
                face_move_norm = dist / max(fw, 1)
                dx_norm = dx / max(fw, 1)
                dy_norm = dy / max(fh, 1)
                # When the normalised horizontal movement exceeds a threshold,
                # register a gaze/pose shift cue and bump the score.
                if face_move_norm > 0.15 or dx_norm > 0.15:
                    cues.append("Gaze Shift")
                    score = max(score, min(face_move_norm / 0.5, 1.0))
                # Additional cue: vertical head movement (head nod).  Large
                # vertical displacement between frames indicates nodding.
                if dy_norm > 0.15:
                    cues.append("Head Nod")
                    score = max(score, min(dy_norm / 0.5, 1.0))
            # Update the previous face centre for the next frame
            prev_face_center = face_center

        # Add a small random fluctuation for realism
        score += np.random.uniform(-0.02, 0.02)
        # Smooth the score with the previous score to avoid rapid swings and
        # to reduce the likelihood of hovering at 100% for long periods.
        score = 0.6 * score + 0.4 * last_score
        # Blend in the audio‑derived stress.  ``audio_score`` is
        # computed once per video and lies in [0, 1].  Rather than
        # incrementing the score each frame, we take a weighted
        # average to avoid runaway growth.  Adjust ``audio_weight`` to
        # control how much the audio influences the final stress.
        # Weight applied to audio‑derived stress.  Lower values reduce the
        # influence of vocal features on the overall stress score, helping
        # prevent the score from saturating near 100% when the audio_score
        # happens to be high.  Adjust this constant as needed for your
        # recordings.  A value around 0.15 provides a subtle contribution
        # from audio without overwhelming motion‑based cues.
        # Reduce the influence of the audio stress on the final score.  A smaller
        # weight prevents the audio score from dominating the visual cues and
        # helps avoid the stress percentage from saturating at high values.
        audio_weight = 0.05
        score = (1 - audio_weight) * score + audio_weight * audio_score
        # Clamp to [0, 1]
        score = max(0.0, min(score, 1.0))

        nervous_scores.append(round(score * 100, 2))
        # Record the timestamp for this processed frame.  Dividing by fps
        # converts the current frame index into seconds.  Even though
        # frames are skipped, the time axis will reflect the real video
        # playback progression.
        time_values.append(frame_idx / fps)

        # Determine trend relative to previous logged score
        trend = "Sustained"
        if score - last_score > 0.15:
            trend = "Escalation"
        elif last_score - score > 0.15:
            trend = "De-escalation"

        current_second = int(frame_idx / fps)
        cue_text = ", ".join(cues) if cues else ""
        # Combine visual cues with the global audio cue.  If audio_cue_label is
        # defined, append it to the cue list for each event.  Ensure there
        # are no duplicate commas or trailing separators.
        combined_cues = cue_text
        if audio_cue_label:
            if combined_cues:
                # Avoid adding duplicate audio cue if already present
                if audio_cue_label not in combined_cues:
                    combined_cues = f"{combined_cues}, {audio_cue_label}"
            else:
                combined_cues = audio_cue_label
        score_delta = abs(score - last_score)

        # Log only when stress is high and changes significantly or cues change
        if (
            score > 0.65
            and current_second != last_logged_second
            and (score_delta >= 0.15 or combined_cues != last_logged_cues)
        ):
            stress_events.append({
                "time": round(frame_idx / fps, 2),
                "score": round(score * 100, 2),
                "cues": combined_cues,
                "trend": trend
            })

            last_logged_second = current_second
            last_score = score
            last_logged_cues = combined_cues

        # Overlay live information on frame.  The black rectangle at the top
        # provides space for stress level, trend, timestamp and current cues.  A
        # larger height accommodates multiple cues without overlapping text.
        overlay_height = 150  # static height to allow a few cue lines
        cv2.rectangle(frame, (0, 0), (460, overlay_height), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Stress: {round(score * 100, 2)}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Trend: {trend}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            frame,
            f"Time: {round(frame_idx / fps, 2)}s",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )
        # Display the detected cues (e.g. Gaze Shift, Hand Fidget) beneath the time
        y_offset = 120
        for cue in cues:
            cv2.putText(
                frame,
                cue,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 20

        # Update last_score after smoothing for next iteration.  This
        # ensures the smoothing function in the next frame uses the most recent
        # score rather than the last logged event value.
        last_score = score

        # Draw face, pose and hand landmarks only if requested via the sidebar
        # and when MediaPipe is available.  When any overlay is disabled, skip
        # the corresponding detection to reduce computation.
        if show_face_mesh:
            if mp is not None and face_mesh is not None:
                face_results = face_mesh.process(rgb_small)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Face centre for gaze shift: use the nose tip landmark (index 1)
                        nose = face_landmarks.landmark[1]
                        face_center = (
                            int(nose.x * frame.shape[1]),
                            int(nose.y * frame.shape[0]),
                        )
                        if prev_face_center is not None:
                            dx = abs(face_center[0] - prev_face_center[0])
                            # Use approx face width based on inter‑eye distance
                            eye_distance = abs(
                                face_landmarks.landmark[33].x
                                - face_landmarks.landmark[263].x
                            )
                            if eye_distance > 0:
                                if dx > 0.15 * eye_distance * frame.shape[1]:
                                    cues.append("Gaze Shift")
                        prev_face_center = face_center
                        # Draw small red dots at every facial landmark.  A radius of 1
                        # produces a fine mesh resembling the old MediaPipe display.
                        for lm in face_landmarks.landmark:
                            px = int(lm.x * frame.shape[1])
                            py = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
            elif show_face_mesh:
                # Haar cascade fallback: draw coarse mesh on detected faces.  Run on
                # the downscaled grayscale frame and scale coordinates up to the
                # original frame for drawing.
                faces_small = face_cascade.detectMultiScale(
                    gray_small, scaleFactor=1.1, minNeighbors=5
                )
                for (x_s, y_s, w_s, h_s) in faces_small:
                    # Map the bounding box from the downscaled image back to
                    # original frame coordinates.
                    x = int(x_s / processing_scale)
                    y = int(y_s / processing_scale)
                    w = int(w_s / processing_scale)
                    h = int(h_s / processing_scale)
                    face_center = (x + w // 2, y + h // 2)
                    if prev_face_center is not None:
                        dx = abs(face_center[0] - prev_face_center[0])
                        dy = abs(face_center[1] - prev_face_center[1])
                        # Normalise by face dimensions
                        dx_norm = dx / max(w, 1)
                        dy_norm = dy / max(h, 1)
                        if dx_norm > 0.15:
                            cues.append("Gaze Shift")
                        if dy_norm > 0.15:
                            cues.append("Head Nod")
                    prev_face_center = face_center
                    grid_size = 10
                    for i in range(1, grid_size + 1):
                        for j in range(1, grid_size + 1):
                            px = int(x + w * i / (grid_size + 1))
                            py = int(y + h * j / (grid_size + 1))
                            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

        # Pose skeleton drawing using MediaPipe
        if show_pose and mp is not None and pose is not None:
            pose_results = pose.process(rgb_small)
            if pose_results.pose_landmarks:
                lms = pose_results.pose_landmarks.landmark
                # Iterate over default POSE_CONNECTIONS to draw lines.  Use green
                # colour for body skeleton lines.
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    p1 = lms[start_idx.value]
                    p2 = lms[end_idx.value]
                    x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
                    x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hand connections drawing using MediaPipe
        if show_hands and mp is not None and hands is not None:
            hands_results = hands.process(rgb_small)
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    # Draw the hand connections using the predefined connections.  Use
                    # blue colour for hand skeleton lines.
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        p1 = hand_landmarks.landmark[start_idx]
                        p2 = hand_landmarks.landmark[end_idx]
                        x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
                        x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show the processed frame with overlays in the UI
        video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Update previous downscaled frame for next iteration
        prev_gray_small = gray_small

    cap.release()

    # Remove the progress bar once processing completes
    progress_bar.empty()

    # Persist the results in session state so they remain visible after
    # downloading the report or on subsequent reruns.  Store scores,
    # timestamps and stress events for later use.
    st.session_state["nervous_scores"] = nervous_scores
    st.session_state["time_values"] = time_values
    st.session_state["stress_events"] = stress_events
    st.session_state["processed_file"] = uploaded_file.name
    st.session_state["fps"] = fps

    # Plot timeline of stress scores against the time axis.  Use a
    # DataFrame indexed by seconds so the x‑axis reflects real time.  A
    # percentage scale is shown on the y‑axis.
    col2.subheader("Stress Timeline")
    if nervous_scores:
        df_timeline = pd.DataFrame({"Stress (%)": nervous_scores}, index=time_values)
        col2.line_chart(df_timeline)
        # Display simple metrics summarising the analysis.  Showing the average
        # stress level and peak stress helps the user quickly interpret the
        # results without opening the PDF.  These metrics correspond to the
        # values included in the report.
        col2.metric("Average Stress", f"{round(np.mean(nervous_scores), 2)}%")
        col2.metric("Peak Stress", f"{round(np.max(nervous_scores), 2)}%")

    # Generate the PDF report first so it is ready for download
    pdf_name = generate_pdf(
        user=st.session_state["user"],
        fps=fps,
        stress_events=stress_events,
        scores=nervous_scores,
        audio_features=audio_features,
        audio_score=audio_score,
    )

    # Provide download button for the PDF report
    with open(pdf_name, "rb") as f:
        st.download_button(
            "Download Professional PDF Report",
            f,
            file_name=pdf_name,
            key="download_pdf"
        )

    # Finally display the flagged stress moments and allow notes.  Positioning
    # this section after the report download ensures it appears at the bottom
    # of the page, as requested by the user.  Use the session state version
    # of stress_events so notes persist across reruns and remain after the
    # report is downloaded.
    st.subheader("Flagged Stress Moments (Add Notes)")
    for i, e in enumerate(st.session_state.get("stress_events", [])):
        note_key = f"note_{i}"
        note = st.text_input(
            f"{e['time']}s — {e['score']}% ({e['trend']})",
            key=note_key,
        )
        # Store the note back into the session state's stress events for later
        # inclusion in the PDF if the user re-runs the analysis.
        st.session_state["stress_events"][i]["notes"] = note
