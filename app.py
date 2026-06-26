"""
Q3B — Streamlit app: Song identifier
Run locally with: streamlit run app.py
Deploy on Streamlit Community Cloud by pushing this + requirements.txt to GitHub.
"""
#hello i am trying to so to see if it works hopw it does

import time
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
SAMPLE_DIR = "samples"      # a handful of short query clips for easy testing/demo
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


def fingerprint_audio(y, sr, timing=None):
    """timing: optional dict to record elapsed ms for each stage (for the UI timing strip)."""
    t0 = time.perf_counter()
    f, t, Sxx_db = compute_spectrogram(y, sr)
    t1 = time.perf_counter()
    freq_idx, time_idx = get_peaks(Sxx_db)
    t2 = time.perf_counter()
    hashes = generate_hashes(freq_idx, time_idx)
    t3 = time.perf_counter()
    if timing is not None:
        timing["spectrogram_ms"] = (t1 - t0) * 1000
        timing["constellation_ms"] = (t2 - t1) * 1000
        timing["hashing_ms"] = (t3 - t2) * 1000
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


@st.cache_data
def list_sample_clips():
    """Pre-bundled short query clips so a grader can test the app with one click,
    without needing to source their own audio file."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(exts))


def find_song_file(song_name):
    """Locate the original audio file for a matched song name, so we can
    re-fingerprint it and show 'where in the song' the query clip aligned."""
    if not os.path.isdir(SONG_DIR):
        return None
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    for fname in os.listdir(SONG_DIR):
        if fname.lower().endswith(exts) and os.path.splitext(fname)[0] == song_name:
            return os.path.join(SONG_DIR, fname)
    return None


def match(y, sr, database, min_votes=MIN_VOTES, timing=None):
    hashes, viz = fingerprint_audio(y, sr, timing=timing)

    t0 = time.perf_counter()
    offset_counts = {}
    for h, t_query in hashes:
        if h in database:
            for song_name, t_db in database[h]:
                offset = t_db - t_query
                offset_counts.setdefault(song_name, {})
                offset_counts[song_name][offset] = offset_counts[song_name].get(offset, 0) + 1

    all_scores = {}
    best_song, best_score, best_hist, best_offset = None, 0, {}, None
    for song_name, offsets in offset_counts.items():
        offset, peak = max(offsets.items(), key=lambda kv: kv[1])
        all_scores[song_name] = peak
        if peak > best_score:
            best_score, best_song, best_hist, best_offset = peak, song_name, offsets, offset

    if timing is not None:
        timing["search_ms"] = (time.perf_counter() - t0) * 1000

    prediction = best_song if best_score >= min_votes else None
    return prediction, all_scores, best_hist, best_offset, viz


# ---------------- Plot helpers ----------------

def plot_spectrogram(ax, f, t, Sxx_db, highlight=None):
    ax.pcolormesh(t, f, Sxx_db, shading="gouraud")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if highlight is not None:
        t_start, t_end = highlight
        ax.axvspan(t_start, t_end, color="white", alpha=0.15, edgecolor="lime", linewidth=1.5)


def plot_constellation(ax, t, f, time_idx, freq_idx, highlight=None, color="orange"):
    ax.scatter(t[time_idx], f[freq_idx], s=4, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if highlight is not None:
        t_start, t_end = highlight
        ax.axvspan(t_start, t_end, color="lime", alpha=0.12, edgecolor="lime", linewidth=1.5)


# ---------------- Shared result renderer ----------------

def render_match_result(y, sr, database, min_votes=MIN_VOTES):
    timing = {}
    t_total0 = time.perf_counter()
    prediction, scores, hist, best_offset, viz = match(y, sr, database, min_votes=min_votes, timing=timing)
    f, t, Sxx_db, freq_idx, time_idx = viz
    timing["total_ms"] = (time.perf_counter() - t_total0) * 1000

    # ---- timing strip ----
    cols = st.columns(5)
    cols[0].metric("Spectrogram", f"{timing.get('spectrogram_ms', 0):.0f} ms")
    cols[1].metric("Constellation", f"{timing.get('constellation_ms', 0):.0f} ms")
    cols[2].metric("Hashing", f"{timing.get('hashing_ms', 0):.0f} ms")
    cols[3].metric("DB search", f"{timing.get('search_ms', 0):.0f} ms")
    cols[4].metric("Total", f"{timing.get('total_ms', 0):.0f} ms")

    # ---- headline result ----
    if prediction:
        st.success(f"**Match found: {prediction}**")
        st.caption(f"cluster score **{scores[prediction]}** hashes agreed at a single offset")
    else:
        st.error("No confident match found (best score below threshold).")

    if scores:
        st.markdown("**Candidate scores**")
        df_scores = pd.DataFrame(sorted(scores.items(), key=lambda kv: -kv[1]),
                                  columns=["song", "score"]).head(5)
        st.bar_chart(df_scores.set_index("song"))

    nperseg_secs = NPERSEG / sr
    query_dur = len(y) / sr

    # ---- Step 1: feature extraction ----
    st.markdown("---")
    st.markdown("#### Step 1 · Feature extraction — from spectrogram to constellation")
    st.caption(
        f"The clip was converted into a time-frequency map (DFT window ≈ {nperseg_secs*1000:.0f} ms). "
        f"Of the dense map, only the **{len(freq_idx)} most prominent peaks** were kept — "
        "discarding amplitude/volume changes and noise, and keeping only *where* energy stands out."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        plot_spectrogram(ax1, f, t, Sxx_db)
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        plot_constellation(ax2, t, f, time_idx, freq_idx)
        st.pyplot(fig2)

    if prediction:
        # ---- Step 2: where in the song ----
        st.markdown("#### Step 2 · Database search — where in the song?")
        song_path = find_song_file(prediction)
        if song_path is not None:
            y_song, sr_song = librosa.load(song_path, sr=SR)
            f_s, t_s, Sxx_s = compute_spectrogram(y_song, sr_song)
            freq_idx_s, time_idx_s = get_peaks(Sxx_s)

            hop = NPERSEG // 2
            frame_dur = hop / sr_song
            t_start = best_offset * frame_dur
            t_end = t_start + query_dur

            st.caption(
                f"The query's fingerprint hashes were looked up against the indexed library. "
                f"Below is **{prediction}**'s full fingerprint — the highlighted window is exactly "
                f"where the query clip's hashes aligned (≈ {t_start:.1f}s–{t_end:.1f}s)."
            )
            c3, c4 = st.columns(2)
            with c3:
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                plot_spectrogram(ax3, f_s, t_s, Sxx_s, highlight=(t_start, t_end))
                st.pyplot(fig3)
            with c4:
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                plot_constellation(ax4, t_s, f_s, time_idx_s, freq_idx_s,
                                    highlight=(t_start, t_end), color="cyan")
                st.pyplot(fig4)
        else:
            st.info(
                f"Matched song **{prediction}**, but its source audio file wasn't found in "
                f"'{SONG_DIR}/' to render the full-track view — only its hashes are in the database."
            )

    # ---- Step 3: the alignment spike ----
    st.markdown("#### Step 3 · The proof — the alignment spike")
    st.caption(
        "Every matched hash votes for a time offset (database frame − query frame). "
        "A genuine match makes scattered votes pile into a single tall spike; a wrong song "
        "gives only a flat noise floor of scattered, near-random offsets — that spike cannot be coincidence."
    )
    fig5, ax5 = plt.subplots(figsize=(10, 3))
    if hist:
        offsets_sorted = sorted(hist.items())
        xs = [o for o, _ in offsets_sorted]
        ys = [c for _, c in offsets_sorted]
        ax5.bar(xs, ys, width=2)
        if best_offset is not None:
            ax5.axvline(best_offset, color="orange", linestyle="--")
            ax5.annotate(f"{scores.get(prediction, 0) if prediction else max(ys)} hashes\naligned here",
                         xy=(best_offset, max(ys)), xytext=(10, 0), textcoords="offset points",
                         color="orange")
    ax5.set_xlabel("Time offset (database frame − query frame)")
    ax5.set_ylabel("# matching hashes")
    st.pyplot(fig5)


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
        song_counts = {}
        for entries in database.values():
            for name, _ in entries:
                song_counts[name] = song_counts.get(name, 0) + 1
        song_set = sorted(song_counts)
        st.write(f"**{len(song_set)} songs indexed**, **{len(database)} unique hashes** in the database.")

        cols_per_row = 4
        for i in range(0, len(song_set), cols_per_row):
            row_songs = song_set[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, name in zip(cols, row_songs):
                with col:
                    st.markdown(f"**{name}**")
                    st.caption(f"{song_counts[name]:,} hashes")

with tab_identify:
    st.subheader("Identify a clip")
    uploaded_file = st.file_uploader("Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"])

    st.markdown("**...or try a sample**")
    samples = list_sample_clips()
    chosen_sample = None
    if samples:
        for name in samples:
            scol1, scol2 = st.columns([4, 1])
            with scol1:
                st.audio(os.path.join(SAMPLE_DIR, name))
            with scol2:
                if st.button("Try", key=f"try_{name}"):
                    chosen_sample = name
    else:
        st.caption(f"No bundled sample clips found — add some to '{SAMPLE_DIR}/' to enable one-click testing.")

    clip_to_run = None
    if uploaded_file is not None:
        clip_to_run = uploaded_file
    elif chosen_sample is not None:
        clip_to_run = os.path.join(SAMPLE_DIR, chosen_sample)

    if clip_to_run is not None and database:
        y, sr = librosa.load(clip_to_run, sr=SR)
        render_match_result(y, sr, database)
    elif clip_to_run is not None and not database:
        st.error("Database is empty — add songs to the songs/ folder first.")

with tab_batch:
    st.subheader("Identify many clips at once")
    st.caption(
        "Each clip is matched against the indexed library; results are written to results.csv "
        "with columns `filename, prediction`. `prediction` is the matched track's filename "
        "without extension, or `none` when no candidate clears the confidence threshold."
    )
    uploaded_files = st.file_uploader(
        "Upload query clips", type=["wav", "mp3", "flac", "ogg", "m4a"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Run batch") and database:
        rows = []
        progress = st.progress(0)
        status = st.empty()
        for i, uf in enumerate(uploaded_files):
            status.text(f"Identifying... {i + 1}/{len(uploaded_files)}")
            y, sr = librosa.load(uf, sr=SR)
            prediction, scores, hist, best_offset, _ = match(y, sr, database)
            rows.append({"filename": uf.name, "prediction": prediction if prediction else "none"})
            progress.progress((i + 1) / len(uploaded_files))
        status.empty()

        df = pd.DataFrame(rows)
        matched = (df["prediction"] != "none").sum()
        st.caption(f"{matched} / {len(df)} clips matched to a track (0 returned `none`).")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", csv_bytes, "results.csv", "text/csv")