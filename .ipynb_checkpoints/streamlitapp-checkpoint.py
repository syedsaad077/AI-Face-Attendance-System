import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
import plotly.express as px
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

if "marked_today" not in st.session_state:
    st.session_state.marked_today = set()

def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = file.read()
    encoded = base64.b64encode(encoded).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/gif;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(page_title="AI Attendance System", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

div[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
}


button[kind="primary"] {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ===================== ADMIN LOGIN =====================
ADMIN_USER = "admin"
ADMIN_PASS = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    set_bg("assets/admin.gif")

    # Remove default padding
    st.markdown("""
        <style>
        .block-container {
            padding-top: 16rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create 3 column layout to center box
    left, center, right = st.columns([4, 2, 4])

    with center:

        st.markdown("""
        <style>
        
        .login-title {
            font-size: 30px;
            margin-bottom: 30px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)

        user = st.text_input("Username", key="user")
        password = st.text_input("Password", type="password", key="pass")

        if st.button("Login", use_container_width=True):
            if user == ADMIN_USER and password == ADMIN_PASS:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()
    
# ===================== SIDEBAR =====================
st.sidebar.title("📌 Navigation")

# 🔴 Sign Out Button
if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.camera_on = False
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["📷 Live Recognition", "➕ Add New Person", "📊 Dashboard", "🗑 Manage Users"]
)

dataset_path = "dataset"
attendance_file = "attendance.xlsx"

# ===================== LOAD KNOWN FACES =====================
@st.cache_resource
def load_faces():
    encodings = []
    names = []

    if os.path.exists(dataset_path):
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)

            if not os.path.isdir(person_folder):
                continue

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                image = face_recognition.load_image_file(img_path)
                face_enc = face_recognition.face_encodings(image)

                if len(face_enc) > 0:
                    encodings.append(face_enc[0])
                    names.append(person_name)

    return encodings, names

known_face_encodings, known_face_names = load_faces()

# ===================== ATTENDANCE FUNCTION =====================
def mark_attendance(name):

    # Stop duplicate marking in same session
    if name in st.session_state.marked_today:
        return

    st.session_state.marked_today.add(name)

    date_today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    new_row = pd.DataFrame([[name, date_today, time_now]],
                           columns=["Name", "Date", "Time"])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(attendance_file, index=False)

# ==============================================================
# ===================== LIVE RECOGNITION =======================
# ==============================================================

if page == "📷 Live Recognition":

    set_bg("assets/live.gif")   # ⚠ Use static image, NOT gif
    

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

   # One master center layout
    left, center, right = st.columns([5,6,5])
    
    with center:
    
        # Heading
        st.markdown(
            "<h1 style='text-align:center;'>📷 Live Face Recognition</h1>",
            unsafe_allow_html=True
        )
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        # Buttons in same width
        btn1, btn2 = st.columns(2)
    
        with btn1:
            if st.button("▶ Start Camera", use_container_width=True):
                st.session_state.camera_on = True
    
        with btn2:
            if st.button("⏹ Stop Camera", use_container_width=True):
                st.session_state.camera_on = False
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        # Camera placeholder
        frame_placeholder = st.empty()

    if st.session_state.camera_on:

        cap = cv2.VideoCapture(0)

        # 🔥 Lower camera resolution (BIG performance boost)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0

        while st.session_state.camera_on:

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 Process every 3rd frame only (massive CPU drop)
            if frame_count % 3 != 0:
                frame_placeholder.image(frame, channels="BGR")
                continue

            # 🔥 Resize smaller for face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_small,
                model="hog"
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small,
                face_locations
            )

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):

                # scale back up
                scale = 1 / 0.4
                top = int(top * scale)
                right = int(right * scale)
                bottom = int(bottom * scale)
                left = int(left * scale)

                name = "Unknown"

                if len(known_face_encodings) > 0:
                    distances = face_recognition.face_distance(
                        known_face_encodings,
                        face_encoding
                    )

                    best_match = np.argmin(distances)

                    if distances[best_match] < 0.5:
                        name = known_face_names[best_match]
                        mark_attendance(name)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            frame_placeholder.image(frame, channels="BGR")

            # 🔥 Reduce FPS slightly (important)
            time.sleep(0.05)

        cap.release()
# ================= ADD PERSON =================
elif page == "➕ Add New Person":
    ...

    set_bg("assets/add.gif")

    st.markdown("""
    <h1 style='text-align:center; margin-left:0%; font-weight:700;'>
    ➕ Capture New Face
    </h1>
    """, unsafe_allow_html=True)

    new_name = st.text_input("Enter Person Name")

    if st.button("Capture Face"):
        if new_name.strip() == "":
            st.warning("Enter valid name")
        else:
            person_folder = os.path.join(dataset_path, new_name)
            os.makedirs(person_folder, exist_ok=True)

            cap = cv2.VideoCapture(0)
            count = 0

            st.info("Capturing 20 images...")

            while count < 20:
                ret, frame = cap.read()
                if not ret:
                    break

                file_path = os.path.join(person_folder, f"{count}.jpg")
                cv2.imwrite(file_path, frame)
                count += 1

            cap.release()
            st.success("Face Captured Successfully ✅")

# ==============================================================
# ===================== DASHBOARD ===============================
# ==============================================================

elif page == "📊 Dashboard":

    set_bg("assets/dashboard.jpeg")

    st.markdown("""
    <h1 style='text-align:center; margin-left:0%; font-weight:700;'>
    📊 Attendance Dashboard
    </h1>
    """, unsafe_allow_html=True)

    if os.path.exists(attendance_file):
        df = pd.read_excel(attendance_file)

        # Date Filter
        df["Date"] = pd.to_datetime(df["Date"])
        min_date = df["Date"].min()
        max_date = df["Date"].max()

        start_date = st.date_input("Start Date", min_date)
        end_date = st.date_input("End Date", max_date)

        filtered_df = df[
            (df["Date"] >= pd.to_datetime(start_date)) &
            (df["Date"] <= pd.to_datetime(end_date))
        ]

        st.dataframe(filtered_df, use_container_width=True)

        # CSV Download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Filtered CSV",
            data=csv,
            file_name="filtered_attendance.csv",
            mime="text/csv"
        )

        # Chart
        fig = px.histogram(filtered_df, x="Name", title="Attendance Count")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No attendance records found.")

# ==============================================================
# ===================== DELETE USER ============================
# ==============================================================

elif page == "🗑 Manage Users":

    set_bg("assets/dashboard.jpeg")

    st.markdown("""
    <h1 style='text-align:center; margin-left:0%; font-weight:700;'>
    🗑 Manage Users
    </h1>
    """, unsafe_allow_html=True)

    if os.path.exists(dataset_path):
        users = [u for u in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, u))]

        if users:
            selected_user = st.selectbox("Select User to Delete", users)

            if st.button("Delete User"):
                import shutil
                shutil.rmtree(os.path.join(dataset_path, selected_user))
                st.success(f"{selected_user} deleted successfully ✅")
                st.rerun()
        else:
            st.info("No users found.")
    else:
        st.info("Dataset folder not found.")