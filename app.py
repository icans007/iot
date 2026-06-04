# ============================================================
# app.py — Tahap 10: Aplikasi Streamlit
# IoT Vulnerability Detection System
# ============================================================
# Pipeline dimuat UTUH dari pipeline_terbaik.pkl.
# Semua input melewati: Scaler → Feature Selection → Model
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================
# CONFIG & CUSTOM CSS (Meningkatkan Tampilan UI)
# ============================================================

st.set_page_config(
    page_title="IoT Vulnerability Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Naikkan batas upload ke 1GB
st._config.set_option("server.maxUploadSize", 1024)

# Injeksi CSS Lanjutan untuk UI yang lebih clean, modern, dan menarik
st.markdown("""
<style>
    /* Import Font Modern */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Menyembunyikan menu default dan footer Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Judul Aplikasi dengan Efek Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        padding-bottom: 10px;
    }
    
    /* Styling Tombol Primary (Gradient, Shadow, Hover Effect) */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        color: white;
        border: none;
    }
    
    /* Styling khusus untuk metrik di sidebar agar lebih menonjol */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #2a5298 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #555 !important;
    }
    
    /* Merapikan Expander agar terlihat seperti Card modern */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        border: 1px solid #e9ecef;
        transition: background-color 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #e2e8f0;
    }
    div[data-testid="stExpanderDetails"] {
        border: 1px solid #e9ecef;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }
    
    /* Styling untuk Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #2a5298 !important;
        color: #2a5298 !important;
    }
    
    /* Input Number styling */
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #ccc;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 1px #2a5298;
    }

    /* Mengatur ulang padding container utama */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DAFTAR 87 KOLOM FITUR (urutan sesuai training)
# ============================================================

ALL_FEATURES = [
    'dur', 'Protocol', 'Length', 'Source Host', 'Destination Host',
    'Sender IP address', 'Target IP address', 'Opcode', 'checksom(ICMP)',
    'Sequence Number (LE)', 'Sequence Number (BE)', 'File Data',
    'Content length', 'Request URI Query', 'Request Method',
    'Full Request URI', 'Request Version', 'Response', 'Ack No',
    'Ack No (RAW)', 'Checksum(TCP)', 'Connection Finish ',
    'Connection Reset', 'Connection Establish Request',
    'Connection Establish Ack', 'Source Port', 'Destination Port',
    'TCPFlags', 'Acknowledgment', 'TCP Segment Length', 'TCP Options',
    'TCP Payload', 'TCP Seq No', 'Src or Drc port', 'Stream index',
    'Time since previous frame', 'Query Name', 'DNS retransmission',
    'DNS query retransmission', 'DNS query retransmission in',
    'LG bit', 'IG bit', 'LG bit.1', 'Duplicate IP address configured',
    'Time to Live', 'Conversation completeness', 'Push', 'Content Type',
    'This is an ACK to the segment in frame', 'ECN-Echo', 'Mode',
    'Type', 'Type.1', 'Window', 'Echo data', 'Accept', 'Status Code',
    'Transaction ID', 'Handshake Type', 'Flags', 'Packet Type',
    'MSS Value', 'Message type', 'Timestamp value', 'TSecr',
    'TCP Option - SACK permitted', 'Response time', 'No response seen',
    'Kind', 'Duplicate ACK #',
    'This frame is a (suspected) retransmission',
    'Previous segment(s) not captured (common at capture start)',
    'FTP Data', 'Bytes in flight', 'Request command', 'Request command.1',
    'BER Error: length is not valid', 'Length.1', 'Flags.1',
    'Packet Length (encrypted)', 'Direction',
    'TCP Option - Maximum segment size', 'Data', 'Checksum',
    'CDATA', 'Label', 'Attack_Category'
]

# 10 fitur yang ditampilkan di input manual
TOP_FEATURES = [
    'dur', 'Length', 'Source Port', 'Destination Port',
    'Time to Live', 'TCP Segment Length', 'TCP Payload',
    'Bytes in flight', 'Acknowledgment', 'TCP Seq No'
]

# Label serangan (kelas Attack_sub_category setelah LabelEncoder)
LABEL_MAP = {
    0: "ARPPoisoning",
    1: "Backdoor",
    2: "ICMPflood",
    3: "ICMPredirect",
    4: "Normal",
    5: "Password_crack",
    6: "Port_Scanning",
    7: "SQLInjection",
    8: "Smurf",
    9: "SYN_FLOOD",
    10: "UDP_flood",
    11: "VUlnerability_Scan"
}

# ============================================================
# LOAD PIPELINE UTUH
# ============================================================

@st.cache_resource
def load_pipeline():
    """
    Memuat Pipeline secara utuh dari pipeline_terbaik.pkl.
    Pipeline mencakup: StandardScaler → SelectFromModel → RandomForest.
    Input baru otomatis diproses lengkap tanpa preprocessing manual.
    """
    try:
        pipeline = joblib.load("pipeline_terbaik.pkl")
        return pipeline, None
    except FileNotFoundError:
        return None, "File 'pipeline_terbaik.pkl' tidak ditemukan. Jalankan mlflow.py terlebih dahulu."
    except Exception as e:
        return None, str(e)

model, load_error = load_pipeline()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ IoT Security")
    st.markdown("---")

    st.markdown("**Dataset**")
    st.caption("IoT Vulnerability Dataset (Preprocessed Balanced)")

    st.markdown("**Feature Selection**")
    st.caption("Embedded Method — SelectFromModel")

    st.markdown("**Classifier**")
    st.caption("Random Forest Classifier")

    st.markdown("**Cross Validation**")
    st.caption("Stratified K-Fold (5 Fold)")

    st.markdown("---")
    st.markdown("**📈 Model Performance**")
    st.metric("Accuracy",  "90.03%")
    st.metric("F1 Score",  "86.71%")
    st.metric("Precision", "87.45%")
    st.metric("Recall",    "86.71%")

    st.markdown("---")

    # Info pipeline step
    st.markdown("**🔗 Urutan Pipeline**")
    st.caption("1. StandardScaler")
    st.caption("2. SelectFromModel")
    st.caption("3. RandomForestClassifier")

    st.markdown("---")
    st.caption("Mid Semester — Machine Learning II")

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ IoT Vulnerability Detection")
st.caption(
    "Sistem deteksi serangan trafik IoT menggunakan Machine Learning Pipeline. "
    "Pipeline dimuat utuh: Scaler → Feature Selection → Model."
)

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()
else:
    st.success(
        "✅ **pipeline_terbaik.pkl** berhasil dimuat  |  "
        "Pipeline: StandardScaler → SelectFromModel → RandomForestClassifier"
    )

st.divider()

# ============================================================
# SECTION 1 — INFO PIPELINE (3 CARD KLIKABLE)
# ============================================================

st.subheader("ℹ️ Informasi Pipeline & Metode")
st.caption("Klik setiap kartu untuk melihat detail.")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📊 Balanced Dataset", expanded=False):
        st.markdown("##### IoT Vulnerability Dataset")
        st.markdown("- **Sumber:** Mendeley Data")
        st.markdown("- **Format:** Preprocessed Balanced CSV")
        st.markdown("- **Jumlah fitur:** 87 kolom")
        st.markdown("- **Target:** `Attack_sub_category` (12 kelas)")
        st.markdown("- **Kelas:** Normal, ARPPoisoning, Backdoor, ICMPflood, ICMPredirect, Password_crack, Port_Scanning, SQLInjection, Smurf, SYN_FLOOD, UDP_flood, VUlnerability_Scan")
        st.markdown("- Dataset di-*balance* agar distribusi tiap kelas proporsional")

with col2:
    with st.expander("🌲 Random Forest", expanded=False):
        st.markdown("##### Random Forest Classifier")
        st.markdown("- **Tipe:** Ensemble — kumpulan Decision Tree")
        st.markdown("- **n_estimators:** 100 pohon")
        st.markdown("- **max_depth:** 10")
        st.markdown("- **min_samples_split:** 2")
        st.markdown("- **min_samples_leaf:** 1")
        st.markdown("- **Tuning:** GridSearchCV / RandomizedSearchCV")
        st.markdown("- **Validasi:** Stratified K-Fold (5 Fold)")
        st.markdown("- Mendukung `predict_proba` → probabilitas tiap kelas")

with col3:
    with st.expander("⚙️ Embedded Feature Selection", expanded=False):
        st.markdown("##### SelectFromModel (Embedded Method)")
        st.markdown("- Seleksi fitur dari **87 → 15 fitur terbaik**")
        st.markdown("- Berdasarkan *feature importance* dari Random Forest")
        st.markdown("- Terintegrasi **di dalam Pipeline** → No Data Leakage")
        st.markdown("- Fitur di bawah threshold importance otomatis dibuang")
        st.markdown("- Perbandingan metode:")
        st.markdown("  - Filter: SelectKBest (f_classif)")
        st.markdown("  - Wrapper: RFE")
        st.markdown("  - **Embedded: SelectFromModel ← Terbaik**")
        st.markdown("- Urutan: `Scaler → SelectFromModel → Random Forest`")

st.divider()

# ============================================================
# SECTION 2 — KELAS YANG DAPAT DIDETEKSI (KLIKABLE)
# ============================================================

with st.expander("🎯 Kelas Serangan yang Dapat Dideteksi (12 Kelas)", expanded=False):
    kelas_col1, kelas_col2, kelas_col3 = st.columns(3)
    kelas_list = list(LABEL_MAP.values())

    for i, kelas in enumerate(kelas_list):
        icon = "🟢" if kelas == "Normal" else "🔴"
        tipe = "Lalu lintas normal" if kelas == "Normal" else "Serangan"
        if i % 3 == 0:
            kelas_col1.markdown(f"{icon} **{kelas}** — {tipe}")
        elif i % 3 == 1:
            kelas_col2.markdown(f"{icon} **{kelas}** — {tipe}")
        else:
            kelas_col3.markdown(f"{icon} **{kelas}** — {tipe}")

st.divider()

# ============================================================
# SECTION 3 — CARA PENGGUNAAN (KLIKABLE)
# ============================================================

with st.expander("📖 Cara Penggunaan Aplikasi", expanded=False):
    st.markdown("""
    **Tab 🧠 Input Manual**
    - Isi nilai 10 fitur utama jaringan IoT secara manual.
    - Fitur lain yang tidak diisi otomatis bernilai 0.
    - Seluruh 87 kolom dikirim ke Pipeline secara lengkap.
    - Tekan tombol **Prediksi** untuk mendapatkan hasil.

    **Tab 📁 Upload CSV**
    - Upload file CSV berisi data jaringan IoT.
    - Kolom CSV harus sama persis dengan fitur training (87 kolom).
    - Kolom `Attack_sub_category` (label) **tidak boleh ada** di CSV.
    - Kolom yang kurang akan otomatis diisi nilai 0.
    - Tekan **Prediksi Dataset** lalu unduh hasilnya.
    - Batas ukuran file: **1 GB**.

    > ⚠️ Pipeline memuat Scaler dan Feature Selection secara otomatis.
    > Tidak diperlukan preprocessing manual sebelum input data ke UI ini.
    """)

st.divider()

# ============================================================
# SECTION 4 — TAB PREDIKSI
# ============================================================

tab1, tab2 = st.tabs(["🧠 Input Manual", "📁 Upload CSV"])

# ----------------------------------------------------------
# TAB 1: INPUT MANUAL
# ----------------------------------------------------------
with tab1:
    st.markdown("### Input Nilai Fitur Jaringan IoT")
    st.caption(
        "Isi 10 fitur utama di bawah. Fitur lain otomatis bernilai 0. "
        "Semua 87 kolom dikirim ke Pipeline."
    )

    input_values = {col: 0.0 for col in ALL_FEATURES}

    col_a, col_b = st.columns(2)
    for i, fname in enumerate(TOP_FEATURES):
        col = col_a if i % 2 == 0 else col_b
        with col:
            input_values[fname] = st.number_input(
                label=fname,
                value=0.0,
                format="%.4f",
                key=f"fi_{fname}"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Lakukan Prediksi", key="btn_manual", use_container_width=True):
        try:
            data_input = pd.DataFrame([input_values])[ALL_FEATURES]
            hasil_kode = model.predict(data_input)[0]

            # Tampilkan nama kelas (bukan angka)
            if hasattr(model, "classes_"):
                hasil_label = str(hasil_kode)
            else:
                hasil_label = LABEL_MAP.get(int(hasil_kode), str(hasil_kode))

            # Tentukan warna notif
            if hasil_label == "Normal":
                st.success(f"🟢 **Hasil Prediksi:** {hasil_label} — Trafik normal, tidak terdeteksi serangan.")
            else:
                st.error(f"🔴 **Hasil Prediksi:** {hasil_label} — Terdeteksi serangan!")

            # Probabilitas
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(data_input)
                proba_df = pd.DataFrame(
                    proba,
                    columns=[str(c) for c in model.classes_]
                ).round(4)

                with st.expander("📊 Lihat Probabilitas per Kelas", expanded=True):
                    st.dataframe(proba_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.caption("Pastikan nama kolom sesuai dengan fitur yang digunakan saat training.")

# ----------------------------------------------------------
# TAB 2: UPLOAD CSV
# ----------------------------------------------------------
with tab2:
    st.markdown("### Prediksi Batch via File CSV")
    st.caption(
        "Upload CSV dengan 87 kolom fitur (tanpa kolom `Attack_sub_category`). "
        "Kolom yang kurang akan diisi nilai 0 secara otomatis. Batas ukuran: **1 GB**."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        key="upload_csv"
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.markdown(f"**Preview Data:** {len(data)} baris · {len(data.columns)} kolom")
            st.dataframe(data.head(), use_container_width=True)

            # Validasi kolom
            missing_cols = [c for c in ALL_FEATURES if c not in data.columns]
            extra_cols   = [c for c in data.columns  if c not in ALL_FEATURES]

            if missing_cols:
                st.warning(
                    f"⚠️ {len(missing_cols)} kolom tidak ditemukan di CSV dan akan diisi 0: "
                    f"{missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}"
                )
            if extra_cols:
                st.info(
                    f"ℹ️ {len(extra_cols)} kolom tidak dikenal akan diabaikan: "
                    f"{extra_cols[:5]}{'...' if len(extra_cols) > 5 else ''}"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔍 Prediksi Dataset", key="btn_csv", use_container_width=True):
                with st.spinner("Memproses melalui Pipeline: Scaler → Feature Selection → Model..."):

                    # Isi kolom yang kurang dengan 0, urutkan sesuai training
                    for col in ALL_FEATURES:
                        if col not in data.columns:
                            data[col] = 0.0
                    data_ordered = data[ALL_FEATURES]

                    prediksi = model.predict(data_ordered)
                    data_hasil = data_ordered.copy()
                    data_hasil["Prediction"] = prediksi

                    if hasattr(model, "predict_proba"):
                        proba_batch = model.predict_proba(data_ordered)
                        for i, c in enumerate(model.classes_):
                            data_hasil[f"Prob_{c}"] = proba_batch[:, i].round(4)

                st.success(f"✅ {len(data_hasil)} baris selesai diprediksi.")

                # Preview hasil
                st.markdown("**Hasil Prediksi (10 baris pertama):**")
                st.dataframe(data_hasil.head(10), use_container_width=True)

                # Distribusi kelas
                with st.expander("📊 Lihat Distribusi Prediksi", expanded=True):
                    dist = (
                        data_hasil["Prediction"]
                        .value_counts()
                        .reset_index()
                    )
                    dist.columns = ["Kelas", "Jumlah"]
                    dist["Persentase"] = (
                        (dist["Jumlah"] / len(data_hasil) * 100)
                        .round(2)
                        .astype(str) + "%"
                    )
                    dist["Jenis"] = dist["Kelas"].apply(
                        lambda x: "Normal" if str(x) == "Normal" else "Serangan"
                    )
                    st.dataframe(dist, use_container_width=True)

                # Download
                csv_out = data_hasil.to_csv(index=False).encode("utf-8")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (.csv)",
                    data=csv_out,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv",
                    key="btn_download",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.caption("Pastikan file CSV valid dan kolomnya sesuai fitur training.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.85em; font-weight: 600; padding: 1rem;'>
        Mid Semester · Machine Learning II · IoT Vulnerability Detection <br>
        <span style='font-weight: 400;'>Pipeline-based (No Data Leakage) · StandardScaler → SelectFromModel → RandomForestClassifier</span>
    </div>
    """, 
    unsafe_allow_html=True
)