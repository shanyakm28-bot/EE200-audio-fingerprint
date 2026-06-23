"""
Q3B — Streamlit app: Song identifier
Run locally with: streamlit run app.py
Deploy on Streamlit Community Cloud by pushing this + requirements.txt to GitHub.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure
import librosa
import os
import pickle
import pandas as pd
import soundfile as sf

SR = 22050
NPERSEG = 2048
MIN_VOTES = 10
SONG_DIR = "songs"          # commit your indexed song library here
DB_PATH = "database.pkl"

st.set_page_config(page_title="EE200 Audio Fingerprinting", layout="wide")


# ---------------- Core fingerprinting functions ----------------

def compute_spectrogram(y, sr, nperseg=NPERSEG, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def get_peaks(Sxx_db, amp_min_db=-40, neighborhood_size=20):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, neighborhood_size // 2)
    local_max = maximum_filter(Sxx_db, footprint=neighborhood) == Sxx_db
    detected_peaks = local_max & (Sxx_db > amp_min_db)
    freq_idx, time_idx = np.where(detected_peaks)
    return freq_idx, time_idx


def generate_hashes(freq_idx, time_idx, fan_out=10, min_dt=1, max_dt=200):
    peaks = sorted(zip(time_idx, freq_idx))
    hashes = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, fan_out + 1):
            if i + j < len(peaks):
                t2, f2 = peaks[i + j]
                dt = t2 - t1
                if min_dt <= dt <= max_dt:
                    h = (int(f1), int(f2), int(dt))
                    hashes.append((h, int(t1)))
    return hashes


def fingerprint_audio(y, sr):
    f, t, Sxx_db = compute_spectrogram(y, sr)
    freq_idx, time_idx = get_peaks(Sxx_db)
    hashes = generate_hashes(freq_idx, time_idx)
    return hashes, (f, t, Sxx_db, freq_idx, time_idx)


@st.cache_resource
def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as fh:
            return pickle.load(fh)

    database = {}
    if not os.path.isdir(SONG_DIR):
        return database

    for fname in sorted(os.listdir(SONG_DIR)):
        if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue
        song_name = os.path.splitext(fname)[0]
        y, sr = librosa.load(os.path.join(SONG_DIR, fname), sr=SR)
        hashes, _ = fingerprint_audio(y, sr)
        for h, t1 in hashes:
            database.setdefault(h, []).append((song_name, t1))

    with open(DB_PATH, "wb") as fh:
        pickle.dump(database, fh)
    return database


def match(y, sr, database, min_votes=MIN_VOTES):
    hashes, viz = fingerprint_audio(y, sr)
    offset_counts = {}
    for h, t_query in hashes:
        if h in database:
            for song_name, t_db in database[h]:
                offset = t_db - t_query
                offset_counts.setdefault(song_name, {})
                offset_counts[song_name][offset] = offset_counts[song_name].get(offset, 0) + 1

    all_scores = {}
    best_song, best_score, best_hist = None, 0, {}
    for song_name, offsets in offset_counts.items():
        offset, peak = max(offsets.items(), key=lambda kv: kv[1])
        all_scores[song_name] = peak
        if peak > best_score:
            best_score, best_song, best_hist = peak, song_name, offsets

    prediction = best_song if best_score >= min_votes else None
    return prediction, all_scores, best_hist, viz


# ---------------- UI ----------------

st.title("EE200: Audio Fingerprinting")
st.caption("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

database = load_database()

tab_library, tab_identify, tab_batch = st.tabs(["Library", "Identify", "Batch"])

with tab_library:
    st.subheader("Indexed song database")
    if not database:
        st.warning(f"No songs found. Put your song files in the '{SONG_DIR}/' folder and reload.")
    else:
        song_set = sorted({name for entries in database.values() for name, _ in entries})
        st.write(f"**{len(song_set)} songs indexed**, **{len(database)} unique hashes** in the database.")
        st.write(song_set)

with tab_identify:
    st.subheader("Identify a clip")
    uploaded_file = st.file_uploader("Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if uploaded_file is not None and database:
        y, sr = librosa.load(uploaded_file, sr=SR)
        prediction, scores, hist, (f, t, Sxx_db, freq_idx, time_idx) = match(y, sr, database)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spectrogram**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.pcolormesh(t, f, Sxx_db, shading="gouraud")
            ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Frequency (Hz)")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Constellation map**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(t[time_idx], f[freq_idx], s=4, color="orange")
            ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Frequency (Hz)")
            st.pyplot(fig2)

        st.markdown("**Offset histogram (alignment spike)**")
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        if hist:
            ax3.bar(hist.keys(), hist.values(), width=2)
        ax3.set_xlabel("Time offset (database frame − query frame)")
        ax3.set_ylabel("# matching hashes")
        st.pyplot(fig3)

        if prediction:
            st.success(f"**Match found: {prediction}**  (score: {scores[prediction]})")
        else:
            st.error("No confident match found (best score below threshold).")

        if scores:
            st.markdown("**Candidate scores**")
            df_scores = pd.DataFrame(sorted(scores.items(), key=lambda kv: -kv[1]),
                                      columns=["song", "score"])
            st.dataframe(df_scores, use_container_width=True)
    elif uploaded_file is not None and not database:
        st.error("Database is empty — add songs to the songs/ folder first.")

with tab_batch:
    st.subheader("Identify many clips at once")
    st.caption("Each clip is matched against the indexed library; results are written to results.csv "
               "with columns `filename, prediction` (prediction is 'none' if no match clears the threshold).")
    uploaded_files = st.file_uploader(
        "Upload query clips", type=["wav", "mp3", "flac", "ogg", "m4a"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Run batch") and database:
        rows = []
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            y, sr = librosa.load(uf, sr=SR)
            prediction, scores, hist, _ = match(y, sr, database)
            rows.append({"filename": uf.name, "prediction": prediction if prediction else "none"})
            progress.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", csv_bytes, "results.csv", "text/csv")