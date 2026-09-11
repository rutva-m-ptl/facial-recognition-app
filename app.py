import os
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import chromadb
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="Facial Recognition & Enterprise Analytics",
    layout="wide"
)

# --- Initialize OpenCV Haar Cascade Classifier ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Initialize ChromaDB in Writable Temporary Directory ---
CHROMA_DATA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

chroma_client = get_chroma_client()

def get_collection():
    return chroma_client.get_or_create_collection(
        name="gallery_faces_analytics",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

# --- Styling & Layout ---
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

threshold = st.sidebar.slider("Match Distance Threshold", 0.10, 0.80, 0.40, 0.05)

# --- Feature Extractor Engine ---
def convert_to_rgb(img_file):
    return Image.open(img_file).convert("RGB")

def extract_face_embedding_and_bbox(img_np):
    """Detects faces using OpenCV Haar Cascades and extracts normalized feature vectors."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    extracted_faces = []
    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        resized = cv2.resize(face_crop, (32, 32))
        embedding = resized.flatten().astype(np.float32)
        embedding /= (np.linalg.norm(embedding) + 1e-6)

        extracted_faces.append({
            "embedding": embedding.tolist(),
            "bbox": {"top": int(y), "right": int(x + w), "bottom": int(y + h), "left": int(x)}
        })

    return extracted_faces

def draw_bounding_box_and_crop(pil_img, facial_area, label="Match"):
    cv_img = np.array(pil_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    top, right, bottom, left = facial_area['top'], facial_area['right'], facial_area['bottom'], facial_area['left']
    crop_img = cv_img[max(0, top):bottom, max(0, left):right]
    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)) if crop_img.size > 0 else pil_img

    cv2.rectangle(cv_img, (left, top), (right, bottom), (2, 132, 199), 2)
    cv2.putText(cv_img, label, (left, max(15, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (2, 132, 199), 2)

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
            
            img_pil = convert_to_rgb(file)
            img_np = np.array(img_pil)

            try:
                faces = extract_face_embedding_and_bbox(img_np)

                for face_idx, face_data in enumerate(faces):
                    unique_id = f"{file.name}_face_{face_idx}"

                    try:
                        collection.upsert(
                            ids=[unique_id],
                            embeddings=[face_data["embedding"]],
                            metadatas=[{
                                "file_name": file.name,
                                "face_idx": face_idx,
                                "top": face_data["bbox"]["top"],
                                "right": face_data["bbox"]["right"],
                                "bottom": face_data["bbox"]["bottom"],
                                "left": face_data["bbox"]["left"]
                            }]
                        )
                        indexed_count += 1
                    except Exception as inner_e:
                        chroma_client.delete_collection("gallery_faces_analytics")
                        collection = get_collection()
                        collection.upsert(
                            ids=[unique_id],
                            embeddings=[face_data["embedding"]],
                            metadatas=[{
                                "file_name": file.name,
                                "face_idx": face_idx,
                                "top": face_data["bbox"]["top"],
                                "right": face_data["bbox"]["right"],
                                "bottom": face_data["bbox"]["bottom"],
                                "left": face_data["bbox"]["left"]
                            }]
                        )
                        indexed_count += 1

            except Exception as e:
                st.error(f"Execution failure on file {file.name}: {str(e)}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()
        st.sidebar.success(f"Indexed {indexed_count} facial features.")

db_count = collection.count()
st.sidebar.info(f"Database Index Records: {db_count}")

if st.sidebar.button("Reset Vector Database", use_container_width=True):
    try:
        chroma_client.delete_collection("gallery_faces_analytics")
    except Exception:
        pass
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
            
            if "captured_image_bytes" in st.session_state and st.session_state["captured_image_bytes"] is not None:
                st.image(st.session_state["captured_image_bytes"], caption="Active Captured Reference", use_container_width=True)
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
            target_bytes = io.BytesIO(st.session_state["captured_image_bytes"])
            target_img = convert_to_rgb(target_bytes)
            target_np = np.array(target_img)

            try:
                faces = extract_face_embedding_and_bbox(target_np)

                if not faces:
                    st.error("Facial feature extraction failed on reference capture.")
                    st.stop()

                target_embedding = faces[0]["embedding"]

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
                            facial_area = {
                                'top': meta.get('top', 0),
                                'right': meta.get('right', 100),
                                'bottom': meta.get('bottom', 100),
                                'left': meta.get('left', 0)
                            }

                            confidence = max(0.0, min(100.0, (1 - dist) * 100))
                            
                            if file_name not in matched_records or confidence > matched_records[file_name]["confidence"]:
                                matched_records[file_name] = {
                                    "confidence": confidence,
                                    "facial_area": facial_area
                                }

                st.session_state["last_search_results"] = matched_records

            except Exception as e:
                st.error(f"Detection pipeline failure: {str(e)}")
                st.stop()

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
                            st.image(annotated_pil, caption=f"File: {file_name}", use_container_width=True)

                        with crop_col:
                            st.write("**Extracted Target Region**")
                            st.image(cropped_face_pil, width=140)
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
                                    st.image(file_dict[fname], caption=fname, use_container_width=True)
                                else:
                                    st.write(f"File: {fname}")