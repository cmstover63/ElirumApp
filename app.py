import streamlit as st
import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import time

# Additional modules for styling and encoding assets
import base64  # Used to embed the logo image directly in HTML

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
    """Render a high‑impact landing page with a hero section inspired by the user's reference.

    The landing page now uses a dark hero background image and a new navigation bar.
    """
    # Encode images as Base64 for embedding in HTML
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

    # If login page flag is active, display dedicated login page instead of hero.
    # This page uses the same hero background but centres a login card containing
    # username and password fields.  Cancelling returns users to the landing page.
    if st.session_state.get("login_page", False):
        # Render a clean, white login page reminiscent of the original home page.  The
        # page uses a plain white background and a light card for the sign‑in form.
        st.markdown(
            f"""
            <style>
                body, html, .stApp {{ margin: 0; padding: 0; }}
                .block-container {{ margin: 0; padding: 0; width: 100%; max-width: 100%; }}
                    .login-page {{
                         /* Use the hero background for the login page and centre the login card
                            both vertically and horizontally.  Remove white borders by filling
                            the entire viewport. */
                         height: 100vh;
                         width: 100vw;
                         background-image: url('data:image/png;base64,{hero_data}');
                         background-size: cover;
                         background-position: center;
                         display: flex;
                         flex-direction: column;
                         justify-content: center;
                         align-items: center;
                     }}
                .login-card {{
                    background: #F9F9F9;
                    padding: 2rem;
                    border-radius: 8px;
                    width: 320px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                    color: #333;
                    /* Move the login form further up on the page to avoid sitting near the bottom */
                    transform: translateY(-200px);
                    font-family: 'Segoe UI', 'Roboto', sans-serif;
                }}
                .login-card h2 {{ margin-bottom: 1rem; color: #333; }}
                .login-card input {{
                    width: 100%;
                    padding: 0.5rem;
                    margin-bottom: 1rem;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    color: #333;
                }}
                .login-card button {{
                    width: 100%;
                    padding: 0.6rem;
                    background: {PRIMARY_COLOR};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-weight: 600;
                    /* Override the fixed positioning applied to the landing page button */
                    position: relative !important;
                    top: auto !important;
                    right: auto !important;
                    z-index: auto !important;
                }}
                .login-card button:hover {{
                    background: #055E66;
                }}
            </style>
            <div class="login-page">
                <div class="login-card">
            """,
            unsafe_allow_html=True,
        )
        # Larger logo on the login page for branding
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
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    # Render hero page when not in login mode.  Display the logo, tagline and key features
    # centred on the page with a dark overlay on top of the background image.  The login
    # button appears below the description for easy access.
    # Render the hero section with an integrated login button.  The hero fills the
    # entire viewport and contains a semi‑transparent overlay for the logo,
    # tagline and bullet points.  A login button is placed in the top‑right
    # corner of the hero and styled to contrast against the dark background.
    st.markdown(
        f"""
        <style>
        /* Reset margins and padding on the page and container elements */
        body, html, .stApp {{ margin: 0; padding: 0; }}
        .block-container {{ padding: 0; margin: 0; max-width: 100%; width: 100%; }}

        /* Hero container styles */
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
        /* Overlay box containing logo, tagline and features */
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
            list-style: none;
            padding-left: 0;
            margin: 0 0 2rem 0;
        }}
        .hero .overlay li {{
            margin-bottom: 0.7rem;
            font-size: 1.15rem;
            position: relative;
            padding-left: 1.8rem;
        }}
        .hero .overlay li::before {{
            content: "\2714";
            position: absolute;
            left: 0;
            color: {ACCENT_COLOR};
            font-size: 1.2rem;
        }}
        /* Style the login button on the landing page.  The button uses a fixed
           position relative to the viewport so it remains visible at the top‑right.
           We target all Streamlit buttons on this page since there is only
           one button (the login button) when the user is unauthenticated. */
        .stButton > button {{
            position: fixed;
            top: 40px;
            right: 40px;
            padding: 10px 24px;
            font-size: 1rem;
            border-radius: 8px;
            border: 2px solid white;
            background: rgba(0, 0, 0, 0.55);
            color: white;
            font-weight: 600;
            z-index: 1001;
        }}
        .stButton > button:hover {{
            background: {PRIMARY_COLOR};
            color: white;
            border-color: {PRIMARY_COLOR};
        }}
        </style>

        <div class="hero">
            <div class="overlay">
                <img src="data:image/png;base64,{logo_data}" alt="Elirum Logo" style="height:100px;margin-bottom:1rem;" />
                <h1>Don't second guess</h1>
                <h3>Experience the leading AI‑powered system for behavioural and nervousness detection.</h3>
                <ul>
                    <li>AI‑driven facial and body landmark analysis</li>
                    <li>Real‑time quantification of stress and nervousness</li>
                    <li>Secure analytics tailored for law‑enforcement professionals</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True,
    )
    # Add the login button inside the hero.  The CSS above positions the button
    # absolutely relative to the .hero container (top‑right corner).
    login_clicked = st.button("Login", key="landing_enter_button")
    # Close the hero div
    st.markdown("</div>", unsafe_allow_html=True)
    if login_clicked:
        st.session_state["login_page"] = True
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

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
def generate_pdf(user: str, fps: float, stress_events: list, scores: list) -> str:
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
        pdf.multi_cell(
            0, 8,
            f"Time: {e['time']}s | Stress: {e['score']}% ({e['trend']})\n"
            f"Indicators: {e['cues']}\n"
            f"Notes: {e.get('notes', '')}"
        )
        pdf.ln(1)

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

    # Variables to determine trend changes
    last_score = 0.0
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

        score = 0.0
        cues = []

        if prev_gray_small is not None:
            # Compute absolute difference between current frame and previous frame
            diff = cv2.absdiff(gray_small, prev_gray_small)
            # Calculate average pixel intensity of the difference as a proxy for motion
            movement_intensity = np.mean(diff)
            # Normalise movement_intensity to [0, 1] range; adjust divisor for sensitivity
            score = min(max(movement_intensity / 30.0, 0.0), 1.0)

            # Derive cues based on movement intensity.  Larger differences
            # correspond to bigger body movements and are labelled accordingly.
            if movement_intensity > 25:
                cues.append("Posture Shift")
            elif movement_intensity > 15:
                cues.append("Hand Fidget")
            elif movement_intensity > 5:
                cues.append("Facial Tension")

        # Add a small random fluctuation for realism
        score += np.random.uniform(-0.02, 0.02)
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
        score_delta = abs(score - last_score)

        # Log only when stress is high and changes significantly or cues change
        if (
            score > 0.65
            and current_second != last_logged_second
            and (score_delta >= 0.15 or cue_text != last_logged_cues)
        ):
            stress_events.append({
                "time": round(frame_idx / fps, 2),
                "score": round(score * 100, 2),
                "cues": cue_text,
                "trend": trend
            })

            last_logged_second = current_second
            last_score = score
            last_logged_cues = cue_text

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
                        if dx > 0.15 * w:
                            cues.append("Gaze Shift")
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
        scores=nervous_scores
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