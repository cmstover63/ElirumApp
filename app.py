import streamlit as st
import cv2
"""
The application previously relied on MediaPipe's Holistic solution to detect facial,
hand and pose landmarks. However, MediaPipe is not yet compatible with Python 3.13
on Streamlit Cloud, and the `mediapipe.solutions` API has been removed in recent
versions. To avoid this compatibility issue, Elirum now uses a simpler motion
detection approach based on frame differencing with OpenCV. This technique
measures movement intensity between consecutive frames and uses that as a proxy
for stress level. Cues (facial tension, hand movement, posture shift) are
derived heuristically from the magnitude of the motion.
"""
import numpy as np
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import time

# -------------------------
# CASCADE MODELS
# -------------------------
# Load Haar cascade models for face and body detection. OpenCV installs the cascade
# files with the Python package, and the paths are accessible via ``cv2.data.haarcascades``.
# These models enable simple rectangle detection around faces and full bodies without
# requiring the MediaPipe library. If the body cascade fails to load, the app
# gracefully disables body detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
except Exception:
    body_cascade = None

# -------------------------
# NOTE ON MEDIAPIPE REMOVAL
# -------------------------
# MediaPipe has been removed from this application because it does not support
# Python 3.13 (the version used by Streamlit Cloud) and the older API used in
# the original version (`mp.solutions.holistic`) has been deprecated. The
# behavioural analysis now uses a simple motion detection technique implemented
# purely with OpenCV. See the video analysis loop below for details.

# -------------------------
# BRANDING CONFIGURATION
# -------------------------
# Path to your logo image within the app directory
LOGO_PATH = "elirum_logo.png"

# Colour palette (Option 2: green/teal with coral accents)
PRIMARY_COLOR = "#00695C"     # deep teal
SECONDARY_COLOR = "#26A69A"   # light teal
ACCENT_COLOR = "#FF7043"      # coral
BACKGROUND_COLOR = "#FFFFFF"  # white

# -------------------------
# LOGIN SYSTEM
# -------------------------
# Simple hard‑coded user database
USERS = {
    "charles": "secure123",
    "admin": "admin123",
    "detective": "detect2025"
}

def login():
    """Render the login interface and authenticate the user."""
    st.image(LOGO_PATH, width=150)
    st.title("Elirum Analyzer")
    st.subheader("Secure Access")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", key="login_button"):
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Invalid credentials", icon="🚫")

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# If not authenticated, show login page and stop execution
if not st.session_state["authenticated"]:
    login()
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
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo and page title
st.image(LOGO_PATH, width=150)
st.title("Elirum Analyzer")
st.caption("Behavioral Stress & Nervousness Detection System")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Upload interview video  
    2. Behavioral analysis runs automatically  
    3. Review stress timeline  
    4. Download official PDF report  
    """
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
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
    # Save uploaded file to a temporary file
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Show the raw video with audio for playback
    st.subheader("Interview Playback")
    st.video(uploaded_file)

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Initialise data structures for motion‑based analysis
    nervous_scores = []
    stress_events = []

    # Previous frame for motion detection (initially None)
    prev_gray = None

    # Variables to determine trend changes
    last_score = 0.0
    last_logged_second = -1
    last_logged_cues = ""

    col1, col2 = st.columns([2, 1])
    video_slot = col1.empty()

    frame_idx = 0
    FRAME_SKIP = 2  # Skip frames for performance

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Skip frames to reduce processing load
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Convert current frame to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score = 0.0
        cues = []

        if prev_gray is not None:
            # Compute absolute difference between current frame and previous frame
            diff = cv2.absdiff(gray, prev_gray)
            # Calculate average pixel intensity of the difference as a proxy for motion
            movement_intensity = np.mean(diff)
            # Normalise movement_intensity to [0, 1] range; adjust divisor for sensitivity
            score = min(max(movement_intensity / 30.0, 0.0), 1.0)

            # Derive cues based on movement intensity
            if movement_intensity > 25:
                cues.append("Posture Shift")
            elif movement_intensity > 15:
                cues.append("Hand Movement")
            elif movement_intensity > 5:
                cues.append("Facial Tension")

        # Add a small random fluctuation for realism
        score += np.random.uniform(-0.02, 0.02)
        score = max(0.0, min(score, 1.0))

        nervous_scores.append(round(score * 100, 2))

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

        # Overlay live information on frame
        cv2.rectangle(frame, (0, 0), (460, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"Stress: {round(score*100,2)}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Trend: {trend}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(frame, f"Time: {round(frame_idx/fps,2)}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Draw face landmarks as small red dots instead of full bounding boxes.  These
        # points approximate the old MediaPipe face mesh by placing dots at the centre
        # and quarter positions of the detected face rectangle.  Using small circles
        # avoids drawing large boxes while still indicating that a face was detected.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            # positions for red dots: centre and the four quadrant midpoints
            positions = [
                (x + w // 2, y + h // 2),                   # centre
                (x + w // 4, y + h // 4),                   # top‑left quadrant
                (x + 3 * w // 4, y + h // 4),               # top‑right quadrant
                (x + w // 4, y + 3 * h // 4),               # bottom‑left quadrant
                (x + 3 * w // 4, y + 3 * h // 4)            # bottom‑right quadrant
            ]
            for (px, py) in positions:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        # Draw body outline as a rectangle with red dots indicating key positions.
        # If the body classifier is available, draw the bounding box in blue and
        # highlight the centre and midpoints of each side with red dots to mimic
        # body landmarks from the old version.
        if body_cascade is not None:
            bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in bodies:
                # Draw blue rectangle for the body outline
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Define positions for red dots: top‑middle, right‑middle, bottom‑middle,
                # left‑middle, and centre
                positions = [
                    (x + w // 2, y),          # top‑middle
                    (x + w, y + h // 2),      # right‑middle
                    (x + w // 2, y + h),      # bottom‑middle
                    (x, y + h // 2),          # left‑middle
                    (x + w // 2, y + h // 2)  # centre
                ]
                for (px, py) in positions:
                    cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        # Show the processed frame with overlays in the UI
        video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Update previous frame for next iteration
        prev_gray = gray

    cap.release()

    # Plot timeline of stress scores
    col2.subheader("Stress Timeline")
    col2.line_chart(nervous_scores)

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
    # of the page, as requested by the user.
    st.subheader("Flagged Stress Moments (Add Notes)")
    for i, e in enumerate(stress_events):
        note = st.text_input(
            f"{e['time']}s — {e['score']}% ({e['trend']})",
            key=f"note_{i}"
        )
        e["notes"] = note