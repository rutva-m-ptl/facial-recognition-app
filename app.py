import os
import io
import zipfile
import numpy as np
import streamlit as st
import chromadb
import cv2
from PIL import Image
from deepface import DeepFace
from sklearn.cluster import DBSCAN, KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="Facial Recognition & Enterprise Analytics",
    layout="wide"
)

# --- Initialize ChromaDB Vector Store ---
CHROMA_DATA_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

collection = chroma_client.get_or_create_collection(
    name="gallery_faces_analytics",
    metadata={"hnsw:space": "cosine"}
)

# --- Styling & Layout Density ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0284c7;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .meta-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 4px;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Facial Recognition & Enterprise Analytics")
st.caption("Automated facial indexing and similarity analysis platform")
st.markdown("<hr style='margin: 0.5rem 0 1.2rem 0;'>", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")

st.sidebar.subheader("1. Indexing Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload Media Files", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

detector_backend = st.sidebar.selectbox(
    "Detection Engine",
    ["retinaface", "mtcnn", "opencv"],
    index=0
)

age_calibration = st.sidebar.number_input(
    "Age Adjustment Offset (Years)", 
    min_value=-25, 
    max_value=25, 
    value=-5, 
    help="Offset adjustment to normalize neural network age estimation output."
)

threshold = st.sidebar.slider("Match Distance Threshold", 0.10, 0.80, 0.40, 0.05)

st.sidebar.subheader("2. Search Filters")
filter_emotion = st.sidebar.selectbox(
    "Emotion Filter",
    ["All", "happy", "neutral", "sad", "surprise", "fear", "angry", "disgust"]
)

filter_gender = st.sidebar.selectbox(
    "Gender Filter",
    ["All", "Man", "Woman"]
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=1,
    max_value=100,
    value=(10, 80)
)

# --- Helper Functions ---
def convert_to_rgb(img_file):
    img = Image.open(img_file).convert("RGB")
    return img

def draw_bounding_box_and_crop(pil_img, facial_area, label="Match"):
    cv_img = np.array(pil_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

    crop_img = cv_img[max(0, y):y+h, max(0, x):x+w]
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)) if crop_img.size > 0 else pil_img

    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (2, 132, 199), 2)
    cv2.putText(cv_img, label, (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (2, 132, 199), 2)

    annotated_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return annotated_pil, crop_pil

def zip_clustered_images(clusters, uploaded_files):
    file_dict = {f.name: f for f in uploaded_files}
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for person_label, filenames in clusters.items():
            folder_name = person_label.replace(" ", "_")
            for fname in set(filenames):
                if fname in file_dict:
                    file_obj = file_dict[fname]
                    file_obj.seek(0)
                    zip_file.writestr(f"{folder_name}/{fname}", file_obj.read())
                    
    zip_buffer.seek(0)
    return zip_buffer

# --- INDEXING ENGINE ---
if uploaded_files:
    if st.sidebar.button("Index Gallery Media", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        indexed_count = 0
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {file.name}")
            
            temp_path = f"temp_{idx}.jpg"
            img = convert_to_rgb(file)
            img.save(temp_path, quality=95)

            try:
                analysis_results = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['age', 'gender', 'emotion'],
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                if isinstance(analysis_results, dict):
                    analysis_results = [analysis_results]

                objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                for face_idx, obj in enumerate(objs):
                    embedding = obj["embedding"]
                    facial_area = obj.get("facial_area", {'x': 0, 'y': 0, 'w': 100, 'h': 100})
                    unique_id = f"{file.name}_face_{face_idx}"

                    attr = analysis_results[face_idx] if face_idx < len(analysis_results) else {}

                    raw_age = int(attr.get("age", 25))
                    adjusted_age = max(1, raw_age + age_calibration)
                    
                    gender = str(attr.get("dominant_gender", "Unknown"))
                    emotion = str(attr.get("dominant_emotion", "neutral"))

                    collection.upsert(
                        ids=[unique_id],
                        embeddings=[embedding],
                        metadatas=[{
                            "file_name": file.name,
                            "face_idx": face_idx,
                            "age": adjusted_age,
                            "gender": gender,
                            "emotion": emotion,
                            "x": facial_area['x'],
                            "y": facial_area['y'],
                            "w": facial_area['w'],
                            "h": facial_area['h']
                        }]
                    )
                    indexed_count += 1

            except Exception as e:
                st.error(f"Execution failure on file {file.name}: {str(e)}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()
        st.sidebar.success(f"Indexed {indexed_count} facial features.")

db_count = collection.count()
st.sidebar.info(f"Database Index Records: {db_count}")

if st.sidebar.button("Reset Vector Database", use_container_width=True):
    chroma_client.delete_collection("gallery_faces_analytics")
    st.sidebar.warning("Database records cleared.")
    st.rerun()

# --- MAIN NAVIGATION ---
tab1, tab2 = st.tabs(["Identity Search & Query", "Facial Clustering Analysis"])

# ==========================================
# TAB 1: IDENTITY SEARCH
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            st.subheader("Reference Input")
            
            # Check if image is stored in persistent state
            if "captured_image_bytes" in st.session_state and st.session_state["captured_image_bytes"] is not None:
                st.image(st.session_state["captured_image_bytes"], caption="Active Captured Reference", use_column_width=True)
                if st.button("Take New Photo", use_container_width=True):
                    st.session_state["captured_image_bytes"] = None
                    st.rerun()
            else:
                cam_shot = st.camera_input("Capture Reference Image")
                if cam_shot is not None:
                    st.session_state["captured_image_bytes"] = cam_shot.getvalue()
                    st.rerun()

    with col_right:
        with st.container(border=True):
            st.subheader("Search Execution")
            st.write("Execute query to identify target match across persistent database records.")
            st.write("")
            
            has_photo = "captured_image_bytes" in st.session_state and st.session_state["captured_image_bytes"] is not None
            start_search = st.button("Execute Facial Query", use_container_width=True, disabled=not (has_photo and db_count > 0))
            
            if db_count == 0:
                st.info("System notification: Database contains 0 records. Index media prior to querying.")

    if has_photo and (start_search or "last_search_results" in st.session_state):
        if start_search:
            temp_target = "temp_target.jpg"
            target_bytes = io.BytesIO(st.session_state["captured_image_bytes"])
            target_img = convert_to_rgb(target_bytes)
            target_img.save(temp_target, quality=95)

            try:
                target_objs = DeepFace.represent(
                    img_path=temp_target,
                    model_name="Facenet",
                    detector_backend=detector_backend,
                    enforce_detection=False
                )
                
                target_analysis = DeepFace.analyze(
                    img_path=temp_target,
                    actions=['age', 'gender', 'emotion'],
                    detector_backend=detector_backend,
                    enforce_detection=False
                )

                if not target_objs:
                    st.error("Facial feature extraction failed on reference capture.")
                    st.stop()

                target_embedding = target_objs[0]["embedding"]
                
                if target_analysis:
                    t_attr = target_analysis[0] if isinstance(target_analysis, list) else target_analysis
                    calc_target_age = max(1, int(t_attr.get('age', 25)) + age_calibration)
                    st.session_state["profile_attr"] = f"Reference Profile Attributes | Age: {calc_target_age} | Gender: {t_attr.get('dominant_gender')} | Emotion: {t_attr.get('dominant_emotion')}"

                results = collection.query(
                    query_embeddings=[target_embedding],
                    n_results=min(100, db_count),
                    include=["metadatas", "distances"]
                )

                matched_records = {}
                if results and "distances" in results and len(results["distances"][0]) > 0:
                    distances = results["distances"][0]
                    metadatas = results["metadatas"][0]

                    for dist, meta in zip(distances, metadatas):
                        if dist <= threshold:
                            file_name = meta["file_name"]
                            age = meta.get("age", 0)
                            gender = meta.get("gender", "Unknown")
                            emotion = meta.get("emotion", "neutral")
                            facial_area = {
                                'x': meta.get('x', 0),
                                'y': meta.get('y', 0),
                                'w': meta.get('w', 100),
                                'h': meta.get('h', 100)
                            }

                            if filter_emotion != "All" and emotion.lower() != filter_emotion.lower():
                                continue
                            if filter_gender != "All" and gender.lower() != filter_gender.lower():
                                continue
                            if not (age_range[0] <= age <= age_range[1]):
                                continue

                            confidence = max(0.0, min(100.0, (1 - dist) * 100))
                            
                            if file_name not in matched_records or confidence > matched_records[file_name]["confidence"]:
                                matched_records[file_name] = {
                                    "confidence": confidence,
                                    "age": age,
                                    "gender": gender,
                                    "emotion": emotion,
                                    "facial_area": facial_area
                                }

                st.session_state["last_search_results"] = matched_records

            except Exception as e:
                st.error(f"Detection pipeline failure: {str(e)}")
                st.stop()
                
            finally:
                if os.path.exists(temp_target):
                    os.remove(temp_target)

        if "profile_attr" in st.session_state:
            st.info(st.session_state["profile_attr"])

        matched_records = st.session_state.get("last_search_results", {})

        st.subheader("Query Analytics")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{db_count}</div><div class="metric-label">Indexed Records</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(matched_records)}</div><div class="metric-label">Matching Results</div></div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

        if matched_records and uploaded_files:
            st.subheader("Query Results")
            file_dict = {f.name: f for f in uploaded_files}

            for file_name, rec in matched_records.items():
                if file_name in file_dict:
                    file_obj = file_dict[file_name]
                    original_pil = convert_to_rgb(file_obj)

                    annotated_pil, cropped_face_pil = draw_bounding_box_and_crop(
                        original_pil, 
                        rec["facial_area"], 
                        label=f"Match ({rec['confidence']:.1f}%)"
                    )

                    with st.container(border=True):
                        img_col, crop_col = st.columns([2.5, 1], gap="medium")

                        with img_col:
                            st.image(annotated_pil, caption=f"File: {file_name}", use_column_width=True)

                        with crop_col:
                            st.write("**Extracted Target Region**")
                            st.image(cropped_face_pil, width=140)
                            st.markdown(
                                f"""
                                <div style="margin-top: 8px;">
                                    <span class="meta-tag">Age: {rec['age']}</span>
                                    <span class="meta-tag">Gender: {rec['gender']}</span>
                                    <span class="meta-tag">Emotion: {rec['emotion']}</span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            st.write("")
                            st.progress(int(rec["confidence"]) / 100)
                            st.caption(f"Match Similarity: {rec['confidence']:.1f}%")

                            crop_byte_arr = io.BytesIO()
                            cropped_face_pil.save(crop_byte_arr, format='JPEG')
                            st.download_button(
                                label="Export Facial Region",
                                data=crop_byte_arr.getvalue(),
                                file_name=f"export_{file_name}",
                                mime="image/jpeg",
                                use_container_width=True
                            )

        else:
            st.info("Query returned 0 matching records based on active criteria.")

# ==========================================
# TAB 2: FACIAL CLUSTERING ANALYSIS
# ==========================================
with tab2:
    st.subheader("Automated Facial Grouping & Clustering")
    st.caption("Groups gallery index elements using unsupervised spatial feature metrics.")

    if db_count == 0:
        st.info("System notification: Database contains 0 records. Index media prior to executing clustering.")
    else:
        c1, c2 = st.columns([1, 2], gap="medium")
        
        with c1:
            with st.container(border=True):
                st.write("**Clustering Parameters**")
                algorithm = st.selectbox("Algorithm Selection", ["DBSCAN (Density-Based)", "K-Means"])
                
                if algorithm == "DBSCAN (Density-Based)":
                    eps_val = st.slider("Distance Threshold (eps)", 0.20, 0.80, 0.40, 0.05)
                    min_samples = st.slider("Minimum Cluster Size", 1, 5, 1)
                else:
                    n_clusters = st.slider("Cluster Count Target", 2, max(2, min(20, db_count)), 3)

                run_cluster = st.button("Execute Clustering Analysis", use_container_width=True)

        with c2:
            if run_cluster:
                all_data = collection.get(include=["embeddings", "metadatas"])
                embeddings = np.array(all_data["embeddings"])
                metadatas = all_data["metadatas"]

                if len(embeddings) == 0:
                    st.warning("No vector data available for processing.")
                else:
                    if algorithm == "DBSCAN (Density-Based)":
                        clusterer = DBSCAN(eps=eps_val, min_samples=min_samples, metric="cosine")
                        labels = clusterer.fit_predict(embeddings)
                    else:
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = clusterer.fit_predict(embeddings)

                    person_clusters = {}
                    for label, meta in zip(labels, metadatas):
                        person_name = "Unassigned" if label == -1 else f"Group Cluster {label + 1}"

                        if person_name not in person_clusters:
                            person_clusters[person_name] = []
                        person_clusters[person_name].append(meta["file_name"])

                    st.session_state["person_clusters"] = person_clusters
                    st.success(f"Categorized {len(embeddings)} indexed vectors into {len(person_clusters)} clusters.")

            if "person_clusters" in st.session_state:
                person_clusters = st.session_state["person_clusters"]

                if uploaded_files:
                    zip_data = zip_clustered_images(person_clusters, uploaded_files)
                    st.download_button(
                        label="Export Grouped Archives (.ZIP)",
                        data=zip_data,
                        file_name="clustered_exports.zip",
                        mime="application/zip",
                    )

                st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

                file_dict = {f.name: f for f in uploaded_files} if uploaded_files else {}

                for person_name, filenames in person_clusters.items():
                    unique_filenames = list(set(filenames))
                    with st.expander(f"{person_name} ({len(unique_filenames)} records)", expanded=True):
                        cols = st.columns(4)
                        for idx, fname in enumerate(unique_filenames):
                            with cols[idx % 4]:
                                if fname in file_dict:
                                    st.image(file_dict[fname], caption=fname, use_column_width=True)
                                else:
                                    st.write(f"File: {fname}")