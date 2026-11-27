import hashlib
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import sbox_core as sc
import aes_core as ac

st.set_page_config(page_title="S-box ML untuk Enkripsi Citra", page_icon="🔐", layout="wide")

DEFAULT_KEY = "KRIPTOGRAFI-2026"
DEFAULT_IV = "000102030405060708090a0b0c0d0e0f"


# ---------------- Utilitas ----------------
@st.cache_data(show_spinner=False)
def load_default_image():
    return np.array(Image.open("mandrill.png").convert("RGB"))


@st.cache_data(show_spinner=False)
def metrics_cached(sbox_tuple):
    return sc.all_metrics(np.array(sbox_tuple))


@st.cache_data(show_spinner=False)
def comparison_table():
    rows = []
    for nama, S in sc.reference_sboxes().items():
        rows.append({"S-box": nama, **metrics_cached(tuple(int(v) for v in S))})
    return pd.DataFrame(rows)


def digest(arr):
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()


@st.cache_data(show_spinner=False)
def _run_ml(_img, img_digest):
    return sc.construct_sbox_ml(_img)


def run_ml(img):
    return _run_ml(img, digest(img))


@st.cache_data(show_spinner=False)
def dataset_cached(_img, img_digest):
    return sc.dataset_from_image(_img)


@st.cache_data(show_spinner=False)
def optimum_cached():
    return sc.optimum_sac_deviation()


@st.cache_data(show_spinner=False, max_entries=8)
def _encrypt_cached(_img, img_digest, key, iv, sbox_tuple, two_way):
    return ac.encrypt_image(_img, key, iv, list(sbox_tuple), two_way)


def encrypt_cached(img, key, iv, sbox_tuple, two_way):
    # Streamlit hanya mengambil sampel isi array besar saat membuat kunci cache,
    # jadi citra di-hash sendiri agar perubahan satu piksel tetap terdeteksi.
    return _encrypt_cached(img, digest(img), key, iv, sbox_tuple, two_way)


def sbox_df(S):
    df = pd.DataFrame(np.asarray(S).reshape(16, 16))
    df.index.name = "X"
    df.columns.name = "Y"
    return df


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def heatmap(M, title, vmin, vmax, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def histogram_fig(img, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for k, (ax, col, name) in enumerate(zip(axes, ["#d62728", "#2ca02c", "#1f77b4"], "RGB")):
        ax.bar(range(256), np.bincount(img[..., k].ravel(), minlength=256), color=col, width=1.0)
        ax.axhline(img[..., k].size / 256, color="black", ls="--", lw=0.8)
        ax.set_title(f"Kanal {name}", fontsize=9); ax.set_xlim(0, 255)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------- Sidebar: pilih S-box ----------------
st.sidebar.title("🔐 S-box ML")
st.sidebar.caption("Proyek Akhir Kriptografi · Kelompok 1 · UNNES 2026")
halaman = st.sidebar.radio("Halaman", ["Beranda", "Konstruksi S-box (ML)", "Uji S-box", "Enkripsi Citra"])

st.sidebar.divider()
pilihan = st.sidebar.selectbox("S-box yang dipakai", list(sc.reference_sboxes().keys()) + ["Unggah S-box sendiri"])
if pilihan == "Unggah S-box sendiri":
    up = st.sidebar.file_uploader("File S-box (.txt / .csv, 256 angka)", type=["txt", "csv"])
    teks = st.sidebar.text_area("…atau tempel 256 angka (desimal atau 0x..)", height=100)
    try:
        if up is not None and up.name.endswith(".csv"):
            df_up = pd.read_csv(up, index_col=0)
            S_AKTIF = df_up.values.astype(int).ravel() if df_up.shape == (16, 16) else sc.parse_sbox(up.getvalue().decode())
        elif up is not None:
            S_AKTIF = sc.parse_sbox(up.getvalue().decode())
        elif teks.strip():
            S_AKTIF = sc.parse_sbox(teks)
        else:
            S_AKTIF = None
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(str(e))
        S_AKTIF = None
    if S_AKTIF is None:
        st.sidebar.info("Belum ada S-box. Sementara memakai S-box ML kelompok.")
        S_AKTIF = np.array(sc.SBOX_ML)
else:
    S_AKTIF = sc.reference_sboxes()[pilihan]
S_TUPLE = tuple(int(v) for v in S_AKTIF)


# ---------------- Beranda ----------------
if halaman == "Beranda":
    st.title("Konstruksi S-box AES Berbasis Machine Learning untuk Enkripsi Citra")
    st.markdown(
        """
Aplikasi ini menampilkan hasil proyek akhir mata kuliah Kriptografi:

1. **Konstruksi S-box (ML)** — S-box dibangun dengan rumus $S(x) = A \\cdot x^{-1} \\oplus \\texttt{0x63}$ di GF(2⁸)
   (polinomial irreducible `0x1F3`). Matriks affine $A$ dipilih oleh model **Ridge Regression** yang dilatih dari
   7.122 kandidat matriks hasil bit-plane citra mandrill. Algoritma genetika disediakan sebagai pembanding.
2. **Uji S-box** — balance, bijective, nonlinearity, SAC, BIC-NL, BIC-SAC, LAP, DAP, dan fixed point,
   dibandingkan dengan S-box AES dan S-box penelitian acuan.
3. **Enkripsi Citra** — S-box dipasang di AES-128 (mode CBC dua arah), lalu diuji dengan histogram,
   entropy, NPCR, UACI, dan dekripsi.

Pilih halaman dan S-box di panel kiri.
        """
    )
    m = metrics_cached(S_TUPLE)
    st.subheader(f"Ringkasan: {pilihan}")
    c = st.columns(5)
    c[0].metric("Nonlinearity", m["NL"], help="Ideal 112")
    c[1].metric("SAC", m["SAC"], help="Ideal 0,5")
    c[2].metric("BIC-SAC", m["BIC-SAC"], help="Ideal 0,5")
    c[3].metric("LAP", m["LAP"], help="Makin kecil makin baik")
    c[4].metric("DAP", m["DAP"], help="Makin kecil makin baik")
    st.caption("Referensi: Alamsyah, Prasetiyo & Muhammad (2023), Scientific Journal of Informatics 10(2); "
               "Alamsyah dkk. (2025), Nonlinear Dynamics 113(4).")


# ---------------- Konstruksi S-box ----------------
elif halaman == "Konstruksi S-box (ML)":
    st.title("Konstruksi S-box dengan Machine Learning")
    st.markdown(
        """
**Alur:** bit-plane 0–1 citra → potongan 64 bit menjadi kandidat matriks affine 8×8 → hanya matriks invertible
(rank 8 di GF(2)) yang dipakai → setiap kandidat diberi label SAC → model dilatih → bobot model dipakai untuk
menyusun matriks affine dengan deviasi SAC terkecil.

**Fitur bag-of-rows:** bit keluaran ke-*j* hanya ditentukan oleh baris ke-*j* matriks affine, sehingga setiap matriks
direpresentasikan sebagai jumlah kemunculan tiap nilai baris (vektor 256 dimensi).
        """
    )
    sumber = st.radio("Citra sumber dataset", ["Mandrill bawaan", "Unggah citra lain"], horizontal=True)
    img_src = load_default_image()
    if sumber == "Unggah citra lain":
        f = st.file_uploader("Citra (PNG/JPG/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="ml_img")
        if f is not None:
            img_src = np.array(Image.open(f).convert("RGB"))
    st.image(img_src, caption=f"Citra sumber {img_src.shape[1]}×{img_src.shape[0]}", width=220)

    if st.button("Jalankan konstruksi S-box", type="primary"):
        with st.spinner("Membangun dataset, melatih model, dan menyusun matriks affine..."):
            t = time.time()
            st.session_state["ml"] = run_ml(img_src)
            st.session_state["ml_time"] = time.time() - t

    if "ml" in st.session_state:
        r = st.session_state["ml"]
        c = st.columns(3)
        c[0].metric("Kandidat matriks", f"{r['n_kandidat']:,}".replace(",", "."))
        c[1].metric("Matriks invertible (dataset)", f"{r['n_dataset']:,}".replace(",", "."))
        c[2].metric("Waktu", f"{st.session_state['ml_time']:.1f} s")

        st.subheader("Evaluasi model (data uji 20%)")
        st.dataframe(pd.DataFrame(r["evaluasi"], columns=["Model", "R²", "MAE"]).round(6), hide_index=True)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for a, pred, t in [(ax[0], r["pred_rf"], "Random Forest (bit mentah)"), (ax[1], r["pred_ridge"], "Ridge bag-of-rows")]:
            a.scatter(r["y_dev_te"], pred, s=5, alpha=0.5)
            lim = [r["y_dev"].min(), r["y_dev"].max()]
            a.plot(lim, lim, "r--", lw=1)
            a.set_title(t, fontsize=10); a.set_xlabel("Deviasi SAC aktual"); a.set_ylabel("Prediksi")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Matriks affine hasil model")
        st.dataframe(pd.DataFrame(r["A"], columns=[f"k{j}" for j in range(8)]), height=320)
        st.subheader("S-box hasil konstruksi")
        st.dataframe(sbox_df(r["sbox"]), height=600)
        dev = sc.sac(r["sbox"])["deviasi"]
        st.success(f"Deviasi SAC: {dev:.5f} (terbaik di dataset tanpa ML: {r['best_dataset_dev']:.5f}). "
                   + ("Sama dengan S-box ML kelompok." if list(r["sbox"]) == sc.SBOX_ML else ""))
        st.download_button("Unduh S-box (.txt)", ", ".join(map(str, r["sbox"])), file_name="sbox_ml.txt")

    # ---------- Pembanding: algoritma genetika ----------
    st.divider()
    st.header("Pembanding: Algoritma Genetika")
    st.markdown(
        """
Matriks affine juga dicari dengan **algoritma genetika (GA)** sebagai pembanding. Kromosom = matriks affine 8×8,
populasi awal diambil dari dataset mandrill, fitness = deviasi SAC, seleksi turnamen (k = 3), crossover per baris
(peluang 0,9), mutasi flip bit (peluang 1/64), elitisme 2, dan anak yang tidak invertible diperbaiki dengan mutasi ulang.
        """
    )
    c = st.columns(3)
    pop_size = c[0].number_input("Ukuran populasi", 10, 200, 60, step=10)
    generations = c[1].number_input("Jumlah generasi", 10, 500, 150, step=10)
    seed = c[2].number_input("Seed", 0, 9999, 1)
    if st.button("Jalankan algoritma genetika"):
        bar = st.progress(0.0, text="Menjalankan GA...")
        t = time.time()
        init = dataset_cached(img_src, digest(img_src))
        st.session_state["ga"] = sc.genetic_algorithm(init, int(pop_size), int(generations), seed=int(seed),
                                                      progress=lambda p: bar.progress(p, text=f"Generasi {int(p * generations)}/{generations}"))
        st.session_state["ga_time"] = time.time() - t
        bar.empty()

    if "ga" in st.session_state:
        g = st.session_state["ga"]
        dev_ml = sc.sac(sc.SBOX_ML)["deviasi"]
        dev_opt = optimum_cached()
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(range(1, len(g["history"]) + 1), g["history"], label="GA (deviasi SAC terbaik)")
        ax.axhline(dev_ml, color="red", ls="--", lw=1.2, label=f"Ridge (ML) = {dev_ml:.5f}")
        ax.set_xlabel("Generasi"); ax.set_ylabel("Deviasi SAC"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        st.pyplot(fig)
        c = st.columns(4)
        c[0].metric("Deviasi SAC GA", f"{g['deviasi']:.5f}")
        c[1].metric("Deviasi SAC Ridge (ML)", f"{dev_ml:.5f}")
        c[2].metric("Optimum teoretis", f"{dev_opt:.5f}")
        c[3].metric("Waktu GA", f"{st.session_state['ga_time']:.1f} s")
        if abs(g["deviasi"] - dev_opt) < 1e-9:
            st.success("GA pada run ini berhasil mencapai nilai optimum, sama dengan Ridge.")
        else:
            st.info("GA belum mencapai nilai optimum pada run ini. Ridge Regression mencapainya secara langsung "
                    "dan hasilnya selalu sama setiap dijalankan, sedangkan hasil GA bergantung pada seed.")
        with st.expander("Matriks affine dan S-box hasil GA"):
            st.dataframe(pd.DataFrame(g["A"], columns=[f"k{j}" for j in range(8)]))
            st.dataframe(sbox_df(g["sbox"]), height=600)
            m = metrics_cached(tuple(int(v) for v in g["sbox"]))
            st.write({k: m[k] for k in ["Balance", "Bijective", "NL", "SAC", "Deviasi SAC"]})


# ---------------- Uji S-box ----------------
elif halaman == "Uji S-box":
    st.title("Pengujian S-box")
    st.caption(f"S-box aktif: **{pilihan}**")
    S = np.array(S_TUPLE)
    with st.expander("Tabel S-box (16×16)", expanded=False):
        st.dataframe(sbox_df(S), height=600)

    with st.spinner("Menghitung metrik..."):
        bal, nl, s, b = sc.balance(S), sc.nonlinearity(S), sc.sac(S), sc.bic(S)
        lap_v, dap_v, fp = sc.lap(S), sc.dap(S), sc.fixed_points(S)

    c = st.columns(5)
    c[0].metric("Balance", "Ya" if bal["seimbang"] else "Tidak")
    c[1].metric("Bijective", "Ya" if sc.bijective(S) else "Tidak")
    c[2].metric("NL (min)", nl["min"])
    c[3].metric("SAC rata-rata", f"{s['avg']:.5f}")
    c[4].metric("Fixed point", fp)
    c = st.columns(5)
    c[0].metric("BIC-NL (min)", b["nl_min"])
    c[1].metric("BIC-SAC", f"{b['sac_avg']:.5f}")
    c[2].metric("LAP", f"{lap_v:.6f}")
    c[3].metric("DAP", f"{dap_v:.6f}")
    c[4].metric("Deviasi SAC", f"{s['deviasi']:.5f}")

    t1, t2, t3, t4 = st.tabs(["Balance & NL", "SAC", "BIC", "Perbandingan"])
    with t1:
        st.dataframe(pd.DataFrame({"Jumlah bit 0": bal["jumlah_0"], "Jumlah bit 1": bal["jumlah_1"],
                                   "Nonlinearity": nl["per_bit"]}, index=[f"f{j}" for j in range(8)]).T)
        st.caption("Balance terpenuhi jika setiap fungsi boolean keluaran punya 128 bit 0 dan 128 bit 1. NL ideal 112.")
    with t2:
        col1, col2 = st.columns([1, 1])
        col1.pyplot(heatmap(s["matrix"], "Matriks SAC (baris: bit input, kolom: bit output)", 0.4, 0.6))
        col2.dataframe(pd.DataFrame(s["matrix"]).round(4))
        col2.caption(f"Rata-rata {s['avg']:.5f} · rentang {s['min']:.4f}–{s['max']:.4f} · ideal 0,5")
    with t3:
        col1, col2 = st.columns(2)
        col1.pyplot(heatmap(b["nl_matrix"], "BIC-NL", 100, 112, "{:.0f}"))
        col2.pyplot(heatmap(b["sac_matrix"], "BIC-SAC", 0.45, 0.55))
    with t4:
        with st.spinner("Menghitung perbandingan..."):
            st.dataframe(comparison_table(), hide_index=True)
        st.caption("NL, BIC-NL, LAP, dan DAP sama untuk semua S-box berbasis invers GF(2⁸) + transformasi affine. "
                   "Perbedaannya terlihat pada SAC dan BIC-SAC.")


# ---------------- Enkripsi Citra ----------------
else:
    st.title("Enkripsi Citra dengan AES-128")
    st.caption(f"S-box aktif: **{pilihan}** · SubBytes & key expansion memakai S-box ini · MixColumns AES standar")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        f = st.file_uploader("Citra (kosongkan untuk memakai mandrill)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        img = np.array(Image.open(f).convert("RGB")) if f is not None else load_default_image()
        if img.shape[0] * img.shape[1] > 1024 * 1024:
            st.warning("Citra besar akan lama dienkripsi di Python murni. Disarankan ≤ 512×512.")
    with col_b:
        key_txt = st.text_input("Kunci (16 karakter)", DEFAULT_KEY, max_chars=16)
        iv_hex = st.text_input("IV (32 digit heksadesimal)", DEFAULT_IV)
        two_way = st.toggle("CBC dua arah (maju + mundur)", value=True,
                            help="CBC biasa hanya merambatkan perubahan ke blok sesudahnya. Lintasan mundur membuat perubahan satu piksel menyebar ke seluruh citra.")

    try:
        key = key_txt.encode()
        iv = bytes.fromhex(iv_hex)
        assert len(key) == 16, "Kunci harus tepat 16 karakter."
        assert len(iv) == 16, "IV harus 16 byte (32 digit heksadesimal)."
        ac.AES128(key, S_TUPLE)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    if st.button("Enkripsi & uji", type="primary"):
        with st.spinner("Mengenkripsi..."):
            t = time.time()
            c_flat, pad = encrypt_cached(img, key, iv, S_TUPLE, two_way)
            t_enc = time.time() - t
            t = time.time()
            dec = ac.decrypt_image(c_flat, img.shape, key, iv, list(S_TUPLE), two_way)
            t_dec = time.time() - t
        st.session_state["enc"] = dict(img=img, c_flat=c_flat, dec=dec, t_enc=t_enc, t_dec=t_dec,
                                       key=key, iv=iv, two_way=two_way, sbox=S_TUPLE)

    if "enc" in st.session_state:
        r = st.session_state["enc"]
        img, c_img, dec = r["img"], ac.cipher_to_image(r["c_flat"], r["img"].shape), r["dec"]
        c = st.columns(3)
        c[0].image(img, caption="Citra asli", width="stretch")
        c[1].image(c_img, caption="Citra terenkripsi", width="stretch")
        c[2].image(dec, caption="Citra hasil dekripsi", width="stretch")
        sama = np.array_equal(img, dec)
        (st.success if sama else st.error)(
            f"Dekripsi {'identik' if sama else 'TIDAK identik'} dengan citra asli · "
            f"piksel berbeda: {int((img != dec).sum())} · enkripsi {r['t_enc']:.1f} s · dekripsi {r['t_dec']:.1f} s")
        d1, d2 = st.columns(2)
        d1.download_button("Unduh citra terenkripsi (PNG)", to_png(c_img), "citra_terenkripsi.png", "image/png")
        d2.download_button("Unduh citra dekripsi (PNG)", to_png(dec), "citra_dekripsi.png", "image/png")

        t1, t2, t3 = st.tabs(["Histogram", "Entropy & korelasi", "NPCR & UACI"])
        with t1:
            st.pyplot(histogram_fig(img, "Histogram citra asli"))
            st.pyplot(histogram_fig(c_img, "Histogram citra terenkripsi (garis putus-putus = frekuensi ideal)"))
            chi = pd.DataFrame({"Kanal": list("RGB"),
                                "χ² asli": [ac.chi_square(img[..., k]) for k in range(3)],
                                "χ² terenkripsi": [ac.chi_square(c_img[..., k]) for k in range(3)]}).round(2)
            st.dataframe(chi, hide_index=True)
            st.caption("Histogram dianggap seragam jika χ² < 293,2478 (α = 0,05, derajat bebas 255).")
        with t2:
            rows = []
            for k, ch in enumerate("RGB"):
                rows.append({"Kanal": ch, "Entropy asli": ac.entropy(img[..., k]),
                             "Entropy terenkripsi": ac.entropy(c_img[..., k]),
                             "Korelasi H asli": ac.correlation(img[..., k], "horizontal"),
                             "Korelasi H terenkripsi": ac.correlation(c_img[..., k], "horizontal"),
                             "Korelasi V terenkripsi": ac.correlation(c_img[..., k], "vertical"),
                             "Korelasi D terenkripsi": ac.correlation(c_img[..., k], "diagonal")})
            df = pd.DataFrame(rows)
            st.dataframe(df.round(4), hide_index=True)
            st.caption(f"Rata-rata entropy terenkripsi: {df['Entropy terenkripsi'].mean():.4f} (ideal 8). "
                       "Korelasi piksel bertetangga citra terenkripsi idealnya mendekati 0.")
        with t3:
            st.markdown("Satu piksel citra asli dinaikkan 1 (mod 256), lalu kedua citra dienkripsi dengan kunci & IV yang sama.")
            h, w = img.shape[:2]
            c = st.columns(3)
            py = c[0].number_input("Baris piksel", 0, h - 1, 0)
            px = c[1].number_input("Kolom piksel", 0, w - 1, 0)
            ch = c[2].selectbox("Kanal", ["R", "G", "B"])
            if st.button("Hitung NPCR & UACI"):
                img2 = img.copy()
                k = "RGB".index(ch)
                img2[py, px, k] = (int(img2[py, px, k]) + 1) % 256
                with st.spinner("Mengenkripsi citra yang diubah..."):
                    c2_flat, _ = encrypt_cached(img2, r["key"], r["iv"], r["sbox"], r["two_way"])
                c2 = ac.cipher_to_image(c2_flat, img.shape)
                rows = []
                for k2, nm in enumerate("RGB"):
                    n_, u_ = ac.npcr_uaci(c_img[..., k2], c2[..., k2])
                    rows.append({"Kanal": nm, "NPCR (%)": n_, "UACI (%)": u_})
                df = pd.DataFrame(rows)
                df.loc[len(df)] = {"Kanal": "Rata-rata", "NPCR (%)": df["NPCR (%)"].mean(), "UACI (%)": df["UACI (%)"].mean()}
                st.dataframe(df.round(4), hide_index=True)
                st.caption("Ideal: NPCR ≈ 99,6094% dan UACI ≈ 33,4635%. Kriteria Wu dkk. (2011) untuk 512×512, α = 0,05: "
                           "NPCR > 99,5893% dan 33,3730% < UACI < 33,5541%.")
