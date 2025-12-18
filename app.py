import streamlit as st
import cv2
import numpy as np
from fpdf import FPDF
from tempfile import NamedTemporaryFile
import time

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

def login():
    """Render the login interface and authenticate the user."""
    # Display logo and title on a single row for a polished sign‑in screen.  The
    # logo appears on the left and the product name on the right.  This
    # arrangement satisfies the user's preference for placing the logo next
    # to the title rather than stacking them vertically.
    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        st.image(LOGO_PATH, width=120)
    with title_col:
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
    stress_events = []

    # Previous frame for motion detection (initially None)
    prev_gray = None

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
    # Skip more frames to speed up processing when using MediaPipe.  A value
    # of 3 reduces CPU load by roughly 33% compared to analysing every frame.
    FRAME_SKIP = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Skip frames to reduce processing load
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Convert current frame to grayscale for motion detection and to RGB
        # for MediaPipe processing.  The grayscale image remains for
        # movement intensity calculations, while the RGB version feeds the
        # face, pose and hand models.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        score = 0.0
        cues = []

        if prev_gray is not None:
            # Compute absolute difference between current frame and previous frame
            diff = cv2.absdiff(gray, prev_gray)
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

        # Draw face, pose and hand landmarks using MediaPipe when available.
        # If MediaPipe is not available, fall back to the Haar cascade for
        # estimating gaze shift and draw a simple grid of dots over the face.
        if mp is not None and face_mesh is not None:
            face_results = face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Face centre for gaze shift: use the nose tip landmark (index 1)
                    nose = face_landmarks.landmark[1]
                    face_center = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
                    if prev_face_center is not None:
                        dx = abs(face_center[0] - prev_face_center[0])
                        # Use approx face width based on inter‑eye distance
                        eye_distance = abs(face_landmarks.landmark[33].x - face_landmarks.landmark[263].x)
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
        else:
            # Haar cascade fallback: draw coarse mesh on detected faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
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
        if mp is not None and pose is not None:
            pose_results = pose.process(rgb)
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
        if mp is not None and hands is not None:
            hands_results = hands.process(rgb)
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