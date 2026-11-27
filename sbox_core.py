"""Konstruksi S-box berbasis machine learning dan metrik pengujian S-box.

Mengikuti Alamsyah dkk. (2023): S(x) = A * x^-1 (+) C di GF(2^8) dengan
polinomial irreducible x^8+x^7+x^6+x^5+x^4+x+1 (0x1F3) dan C = 0x63.
Matriks affine A dipilih oleh model Ridge Regression yang dilatih dari
dataset matriks hasil bit-plane citra mandrill.
"""
import itertools
import numpy as np

POLY = 0x1F3
C_AES = np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int64)  # 0x63, urutan b0..b7
BITS = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.int64)
POW2 = 1 << np.arange(8)

# S-box hasil konstruksi kelompok (hasil notebook Colab)
SBOX_ML = [
    99, 132, 136, 32, 141, 75, 203, 96, 201, 244, 227, 237, 37, 68, 95, 148,
    144, 105, 200, 93, 165, 219, 21, 232, 143, 176, 218, 223, 178, 147, 20, 130,
    217, 89, 30, 88, 25, 110, 7, 235, 73, 224, 185, 117, 91, 191, 81, 100,
    124, 167, 145, 241, 48, 164, 116, 31, 36, 125, 229, 123, 210, 156, 240, 170,
    12, 174, 202, 107, 175, 15, 67, 151, 94, 206, 239, 173, 114, 212, 109, 60,
    86, 216, 153, 190, 208, 49, 135, 84, 127, 119, 168, 196, 2, 53, 146, 41,
    198, 181, 252, 177, 80, 74, 140, 34, 87, 234, 192, 23, 14, 231, 38, 115,
    6, 225, 79, 76, 221, 101, 55, 9, 248, 85, 220, 131, 5, 83, 112, 215,
    134, 65, 189, 47, 172, 150, 171, 35, 52, 4, 186, 90, 43, 162, 40, 56,
    59, 182, 97, 120, 160, 22, 129, 226, 118, 71, 128, 149, 211, 10, 82, 26,
    243, 58, 133, 155, 152, 142, 33, 253, 77, 39, 222, 92, 180, 102, 70, 111,
    250, 42, 50, 57, 197, 45, 28, 1, 54, 245, 19, 17, 108, 137, 138, 66,
    169, 230, 213, 187, 0, 157, 24, 69, 139, 254, 106, 113, 64, 98, 126, 62,
    122, 233, 228, 163, 209, 11, 238, 214, 51, 103, 104, 13, 179, 44, 255, 204,
    251, 78, 16, 154, 46, 246, 18, 236, 193, 184, 27, 63, 166, 159, 194, 3,
    205, 158, 207, 61, 72, 242, 121, 29, 199, 161, 183, 8, 195, 247, 188, 249,
]

A_ML = [[1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1]]

# Matriks affine penelitian acuan (Alamsyah dkk., 2023)
A0 = [[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,1,1,0,0,0],
      [0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[1,0,0,0,0,0,1,1]]
A1 = [[0,1,1,1,1,0,1,0],[0,0,1,1,1,1,0,1],[1,0,0,1,1,1,1,0],[0,1,0,0,1,1,1,1],
      [1,0,1,0,0,1,1,1],[1,1,0,1,0,0,1,1],[1,1,1,0,1,0,0,1],[1,1,1,1,0,1,0,0]]
A2 = [[1,1,1,0,1,0,0,1],[1,1,1,1,0,1,0,0],[0,1,1,1,1,0,1,0],[0,0,1,1,1,1,0,1],
      [1,0,0,1,1,1,1,0],[0,1,0,0,1,1,1,1],[1,0,1,0,0,1,1,1],[1,1,0,1,0,0,1,1]]
A_AES = [[1 if (c - r) % 8 in (0, 4, 5, 6, 7) else 0 for c in range(8)] for r in range(8)]


# ---------------- GF(2^8) ----------------
def gf_mul(a, b, poly=POLY):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def gf_inverse_table(poly=POLY):
    inv = np.zeros(256, dtype=np.int64)
    for a in range(1, 256):
        for b in range(1, 256):
            if gf_mul(a, b, poly) == 1:
                inv[a] = b
                break
    return inv


INV = gf_inverse_table(POLY)
INV_AES = gf_inverse_table(0x11B)


def build_sbox(A, inv=INV, C=C_AES):
    """S(x) = A * inv(x) + C (mod 2). Baris ke-j matriks A menghasilkan bit b_j."""
    y = (BITS[inv] @ np.asarray(A, dtype=np.int64).T + C) % 2
    return (y @ POW2).astype(np.int64)


def gf2_rank(M):
    M = np.array(M, dtype=np.uint8) % 2
    r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def reference_sboxes():
    return {
        "S-box ML (kelompok)": np.array(SBOX_ML),
        "AES standar": build_sbox(A_AES, inv=INV_AES),
        "Acuan S-box1 (A0)": build_sbox(A0),
        "Acuan S-box2 (A2)": build_sbox(A2),
        "Acuan S-box3 (A1)": build_sbox(A1),
    }


# ---------------- Konstruksi dengan ML ----------------
def bag_of_rows(mats):
    rows = (mats * POW2).sum(2)
    X = np.zeros((len(mats), 256))
    for k in range(8):
        X[np.arange(len(mats)), rows[:, k]] += 1
    return X


def construct_sbox_ml(img, n_top=18, seed=42):
    """Bangun S-box dari citra (array HxWx3 uint8). Mengembalikan dict hasil."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    bitstream = np.concatenate([(img >> p) & 1 for p in (0, 1)]).reshape(-1)
    n = len(bitstream) // 64 * 64
    kandidat = bitstream[:n].reshape(-1, 8, 8).astype(np.int64)
    dataset = np.array([A for A in kandidat if gf2_rank(A) == 8])
    sacs = [sac(build_sbox(A)) for A in dataset]
    y_dev = np.array([s["deviasi"] for s in sacs])
    y_avg = np.array([s["avg"] for s in sacs])

    X_raw = dataset.reshape(len(dataset), -1)
    X_row = bag_of_rows(dataset)
    tr, te = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=seed)
    rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1).fit(X_raw[tr], y_dev[tr])
    m_dev = Ridge(alpha=1e-3, fit_intercept=False).fit(X_row[tr], y_dev[tr])
    m_avg = Ridge(alpha=1e-3, fit_intercept=False).fit(X_row[tr], y_avg[tr])

    evaluasi = [
        ("Random Forest (64 bit mentah) → deviasi SAC",
         r2_score(y_dev[te], rf.predict(X_raw[te])), mean_absolute_error(y_dev[te], rf.predict(X_raw[te]))),
        ("Ridge bag-of-rows → deviasi SAC",
         r2_score(y_dev[te], m_dev.predict(X_row[te])), mean_absolute_error(y_dev[te], m_dev.predict(X_row[te]))),
        ("Ridge bag-of-rows → rata-rata SAC",
         r2_score(y_avg[te], m_avg.predict(X_row[te])), mean_absolute_error(y_avg[te], m_avg.predict(X_row[te]))),
    ]

    w_dev, w_avg = m_dev.coef_ * 8, m_avg.coef_ * 8
    urut = [r for r in np.argsort(w_dev) if r != 0][:n_top]
    terbaik = None
    for combo in itertools.combinations(urut, 8):
        c = list(combo)
        skor = (round(w_dev[c].mean(), 6), abs(w_avg[c].mean() - 0.5))
        if terbaik is not None and skor >= terbaik[0]:
            continue
        if gf2_rank(BITS[c]) == 8:
            terbaik = (skor, c)
    A = BITS[sorted(terbaik[1])]
    return {
        "n_kandidat": len(kandidat), "n_dataset": len(dataset),
        "y_dev": y_dev, "pred_rf": rf.predict(X_raw[te]), "pred_ridge": m_dev.predict(X_row[te]),
        "y_dev_te": y_dev[te], "evaluasi": evaluasi, "A": A, "sbox": build_sbox(A),
        "best_dataset_dev": float(y_dev.min()),
    }


# ---------------- Metrik S-box ----------------
def balance(S):
    S = np.asarray(S)
    satu = [int(((S >> j) & 1).sum()) for j in range(8)]
    return {"jumlah_1": satu, "jumlah_0": [256 - v for v in satu], "seimbang": all(v == 128 for v in satu)}


def bijective(S):
    return len(set(int(v) for v in S)) == 256 and min(S) == 0 and max(S) == 255


def walsh(f):
    w = 1 - 2 * np.asarray(f, dtype=np.int64)
    h = 1
    while h < 256:
        w = w.reshape(-1, 2 * h)
        a, b = w[:, :h].copy(), w[:, h:].copy()
        w[:, :h], w[:, h:] = a + b, a - b
        w = w.reshape(-1)
        h *= 2
    return w


def nl_bool(f):
    return int(128 - np.abs(walsh(f)).max() // 2)


def nonlinearity(S):
    S = np.asarray(S)
    per_bit = [nl_bool((S >> j) & 1) for j in range(8)]
    return {"per_bit": per_bit, "avg": float(np.mean(per_bit)), "min": min(per_bit), "max": max(per_bit)}


def sac_matrix(S):
    S = np.asarray(S)
    x = np.arange(256)
    return np.array([BITS[S ^ S[x ^ (1 << i)]].mean(0) for i in range(8)])


def sac(S):
    M = sac_matrix(S)
    return {"matrix": M, "avg": float(M.mean()), "min": float(M.min()), "max": float(M.max()),
            "deviasi": float(np.abs(M - 0.5).mean())}


def bic(S):
    S = np.asarray(S)
    x = np.arange(256)
    nls, sacs = [], []
    nl_mat = np.full((8, 8), np.nan)
    sac_mat = np.full((8, 8), np.nan)
    for j in range(8):
        for k in range(j + 1, 8):
            f = ((S >> j) ^ (S >> k)) & 1
            nl = nl_bool(f)
            s = float(np.mean([(f ^ f[x ^ (1 << i)]).mean() for i in range(8)]))
            nls.append(nl); sacs.append(s)
            nl_mat[j, k] = nl_mat[k, j] = nl
            sac_mat[j, k] = sac_mat[k, j] = s
    return {"nl_min": min(nls), "nl_avg": float(np.mean(nls)), "sac_avg": float(np.mean(sacs)),
            "nl_matrix": nl_mat, "sac_matrix": sac_mat}


def lap(S):
    """Bias maksimum aproksimasi linear: max |#{a.x = b.S(x)}/256 - 1/2|."""
    S = np.asarray(S)
    mx = 0
    for b in range(1, 256):
        g = BITS[S & b].sum(1) % 2
        mx = max(mx, int(np.abs(walsh(g)).max()))
    return mx / 2 / 256


def dap(S):
    S = np.asarray(S)
    x = np.arange(256)
    mx = max(int(np.bincount(S ^ S[x ^ a], minlength=256).max()) for a in range(1, 256))
    return mx / 256


def fixed_points(S):
    S = np.asarray(S)
    return int(np.sum(S == np.arange(256)))


def all_metrics(S):
    S = np.asarray(S)
    n, s, b = nonlinearity(S), sac(S), bic(S)
    return {
        "Balance": "Ya" if balance(S)["seimbang"] else "Tidak",
        "Bijective": "Ya" if bijective(S) else "Tidak",
        "NL": n["min"], "SAC": round(s["avg"], 5), "Deviasi SAC": round(s["deviasi"], 5),
        "BIC-NL": b["nl_min"], "BIC-SAC": round(b["sac_avg"], 5),
        "LAP": round(lap(S), 6), "DAP": round(dap(S), 6), "Fixed point": fixed_points(S),
    }


def parse_sbox(text):
    vals = [int(t, 16) if t.lower().startswith("0x") else int(t)
            for t in text.replace(",", " ").replace(";", " ").split()]
    if len(vals) != 256:
        raise ValueError(f"S-box harus berisi 256 angka, ditemukan {len(vals)}.")
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError("Semua nilai harus di antara 0 dan 255.")
    return np.array(vals, dtype=np.int64)


# ---------------- Pembanding: algoritma genetika ----------------
def genetic_algorithm(init_mats, pop_size=60, generations=150, elit=2, tour=3, pc=0.9, pm=1 / 64, seed=0,
                      progress=None):
    """Cari matriks affine dengan GA. Fitness = deviasi SAC (seri: |SAC rata-rata - 0,5|)."""
    rng = np.random.default_rng(seed)

    def fit_of(A):
        s = sac(build_sbox(A))
        return (s["deviasi"], abs(s["avg"] - 0.5))

    def repair(A):
        A = A.copy()
        while gf2_rank(A) != 8:
            i, j = rng.integers(0, 8, 2)
            A[i, j] ^= 1
        return A

    def pick(pop, fit):
        idx = rng.choice(len(pop), tour, replace=False)
        return pop[min(idx, key=lambda i: fit[i])]

    pop = [init_mats[i].copy() for i in rng.choice(len(init_mats), pop_size, replace=False)]
    fit = [fit_of(A) for A in pop]
    history = []
    for g in range(generations):
        order = np.argsort([f[0] + f[1] * 1e-3 for f in fit])
        new = [pop[i].copy() for i in order[:elit]]
        while len(new) < pop_size:
            p1, p2 = pick(pop, fit), pick(pop, fit)
            child = np.where((rng.random(8) < 0.5)[:, None], p1, p2) if rng.random() < pc else p1.copy()
            child = child ^ (rng.random((8, 8)) < pm)
            new.append(repair(child.astype(np.int64)))
        pop, fit = new, [fit_of(A) for A in new]
        history.append(min(f[0] for f in fit))
        if progress:
            progress((g + 1) / generations)
    best = min(range(pop_size), key=lambda i: fit[i])
    return {"A": pop[best], "sbox": build_sbox(pop[best]), "deviasi": fit[best][0], "history": history}


def dataset_from_image(img):
    bitstream = np.concatenate([(img >> p) & 1 for p in (0, 1)]).reshape(-1)
    n = len(bitstream) // 64 * 64
    return np.array([A for A in bitstream[:n].reshape(-1, 8, 8).astype(np.int64) if gf2_rank(A) == 8])


def optimum_sac_deviation():
    """Deviasi SAC minimum teoretis untuk konstruksi invers GF(2^8) + affine."""
    x = np.arange(256)
    kontribusi = np.zeros(256)
    for r in range(1, 256):
        kontribusi[r] = np.mean([abs((BITS[(INV ^ INV[x ^ (1 << i)]) & r].sum(1) % 2).mean() - 0.5)
                                 for i in range(8)])
    rows = []
    for r in np.argsort(kontribusi):
        if r == 0:
            continue
        if gf2_rank(BITS[rows + [r]]) == len(rows) + 1:
            rows.append(r)
        if len(rows) == 8:
            break
    return float(kontribusi[rows].mean())
