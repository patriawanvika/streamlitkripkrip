"""AES-128 dengan S-box yang bisa diganti, mode CBC dua arah, dan uji citra.

S-box dipakai di SubBytes dan SubWord (key expansion). MixColumns tetap memakai
GF(2^8) AES standar (0x11B), sesuai FIPS-197. Implementasi memakai T-table agar
enkripsi satu citra 512x512x3 cukup cepat di Python murni.
"""
import numpy as np

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _xt(a):
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def _mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xt(a)
        b >>= 1
    return r


def _rotr(w, n):
    return ((w >> n) | (w << (32 - n))) & 0xFFFFFFFF


class AES128:
    def __init__(self, key: bytes, sbox):
        if len(key) != 16:
            raise ValueError("Kunci AES-128 harus 16 byte.")
        S = [int(v) for v in sbox]
        if sorted(S) != list(range(256)):
            raise ValueError("S-box tidak bijektif, sehingga tidak bisa dipakai untuk enkripsi/dekripsi.")
        Si = [0] * 256
        for i, v in enumerate(S):
            Si[v] = i
        self.S, self.Si = S, Si
        te0 = [(_mul(s, 2) << 24) | (s << 16) | (s << 8) | _mul(s, 3) for s in S]
        td0 = [(_mul(s, 14) << 24) | (_mul(s, 9) << 16) | (_mul(s, 13) << 8) | _mul(s, 11) for s in Si]
        self.Te = [te0, [_rotr(w, 8) for w in te0], [_rotr(w, 16) for w in te0], [_rotr(w, 24) for w in te0]]
        self.Td = [td0, [_rotr(w, 8) for w in td0], [_rotr(w, 16) for w in td0], [_rotr(w, 24) for w in td0]]
        self._expand(key)

    def _expand(self, key):
        S = self.S
        w = [int.from_bytes(key[4 * i:4 * i + 4], "big") for i in range(4)]
        for i in range(4, 44):
            t = w[i - 1]
            if i % 4 == 0:
                t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
                t = (S[t >> 24] << 24) | (S[(t >> 16) & 255] << 16) | (S[(t >> 8) & 255] << 8) | S[t & 255]
                t ^= RCON[i // 4 - 1] << 24
            w.append(w[i - 4] ^ t)
        self.ek = [w[4 * r:4 * r + 4] for r in range(11)]
        Td, S = self.Td, self.S

        def inv_mix(x):
            return (Td[0][S[x >> 24]] ^ Td[1][S[(x >> 16) & 255]]
                    ^ Td[2][S[(x >> 8) & 255]] ^ Td[3][S[x & 255]])
        self.dk = [self.ek[10]] + [[inv_mix(x) for x in self.ek[10 - r]] for r in range(1, 10)] + [self.ek[0]]

    def encrypt_words(self, s0, s1, s2, s3):
        T0, T1, T2, T3 = self.Te
        k = self.ek
        s0 ^= k[0][0]; s1 ^= k[0][1]; s2 ^= k[0][2]; s3 ^= k[0][3]
        for r in range(1, 10):
            kr = k[r]
            t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s3 & 255] ^ kr[0]
            t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s0 & 255] ^ kr[1]
            t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s1 & 255] ^ kr[2]
            t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s2 & 255] ^ kr[3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        S, kr = self.S, k[10]
        return ((S[s0 >> 24] << 24 | S[(s1 >> 16) & 255] << 16 | S[(s2 >> 8) & 255] << 8 | S[s3 & 255]) ^ kr[0],
                (S[s1 >> 24] << 24 | S[(s2 >> 16) & 255] << 16 | S[(s3 >> 8) & 255] << 8 | S[s0 & 255]) ^ kr[1],
                (S[s2 >> 24] << 24 | S[(s3 >> 16) & 255] << 16 | S[(s0 >> 8) & 255] << 8 | S[s1 & 255]) ^ kr[2],
                (S[s3 >> 24] << 24 | S[(s0 >> 16) & 255] << 16 | S[(s1 >> 8) & 255] << 8 | S[s2 & 255]) ^ kr[3])

    def decrypt_words(self, s0, s1, s2, s3):
        T0, T1, T2, T3 = self.Td
        k = self.dk
        s0 ^= k[0][0]; s1 ^= k[0][1]; s2 ^= k[0][2]; s3 ^= k[0][3]
        for r in range(1, 10):
            kr = k[r]
            t0 = T0[s0 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s1 & 255] ^ kr[0]
            t1 = T0[s1 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s2 & 255] ^ kr[1]
            t2 = T0[s2 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s3 & 255] ^ kr[2]
            t3 = T0[s3 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s0 & 255] ^ kr[3]
            s0, s1, s2, s3 = t0, t1, t2, t3
        S, kr = self.Si, k[10]
        return ((S[s0 >> 24] << 24 | S[(s3 >> 16) & 255] << 16 | S[(s2 >> 8) & 255] << 8 | S[s1 & 255]) ^ kr[0],
                (S[s1 >> 24] << 24 | S[(s0 >> 16) & 255] << 16 | S[(s3 >> 8) & 255] << 8 | S[s2 & 255]) ^ kr[1],
                (S[s2 >> 24] << 24 | S[(s1 >> 16) & 255] << 16 | S[(s0 >> 8) & 255] << 8 | S[s3 & 255]) ^ kr[2],
                (S[s3 >> 24] << 24 | S[(s2 >> 16) & 255] << 16 | S[(s1 >> 8) & 255] << 8 | S[s0 & 255]) ^ kr[3])

    def encrypt_block(self, block: bytes) -> bytes:
        w = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(4)]
        return b"".join(x.to_bytes(4, "big") for x in self.encrypt_words(*w))

    def decrypt_block(self, block: bytes) -> bytes:
        w = [int.from_bytes(block[4 * i:4 * i + 4], "big") for i in range(4)]
        return b"".join(x.to_bytes(4, "big") for x in self.decrypt_words(*w))


# ---------------- Mode CBC dua arah ----------------
def _to_words(data: bytes):
    return np.frombuffer(data, dtype=">u4").astype(np.int64).reshape(-1, 4).tolist()


def _from_words(words):
    return np.array(words, dtype=">u4").tobytes()


def _cbc_enc(aes, blocks, iv):
    p0, p1, p2, p3 = iv
    out = []
    enc = aes.encrypt_words
    for b in blocks:
        p0, p1, p2, p3 = enc(b[0] ^ p0, b[1] ^ p1, b[2] ^ p2, b[3] ^ p3)
        out.append((p0, p1, p2, p3))
    return out


def _cbc_dec(aes, blocks, iv):
    prev = iv
    out = []
    dec = aes.decrypt_words
    for b in blocks:
        d = dec(*b)
        out.append((d[0] ^ prev[0], d[1] ^ prev[1], d[2] ^ prev[2], d[3] ^ prev[3]))
        prev = b
    return out


def pad_len(n):
    return (-n) % 16


def encrypt_bytes(data: bytes, key: bytes, iv: bytes, sbox, two_way=True):
    """CBC maju lalu (opsional) CBC mundur pada urutan blok terbalik.
    Data harus kelipatan 16 byte (ukuran citra 512x512x3 sudah pas)."""
    aes = AES128(key, sbox)
    ivw = _to_words(iv)[0]
    blocks = _to_words(data)
    c = _cbc_enc(aes, blocks, ivw)
    if two_way:
        c = _cbc_enc(aes, c[::-1], ivw)[::-1]
    return _from_words(c)


def decrypt_bytes(data: bytes, key: bytes, iv: bytes, sbox, two_way=True):
    aes = AES128(key, sbox)
    ivw = _to_words(iv)[0]
    blocks = _to_words(data)
    if two_way:
        blocks = _cbc_dec(aes, blocks[::-1], ivw)[::-1]
    return _from_words(_cbc_dec(aes, blocks, ivw))


def ecb_encrypt_bytes(data: bytes, key: bytes, sbox):
    aes = AES128(key, sbox)
    return _from_words([aes.encrypt_words(*b) for b in _to_words(data)])


def encrypt_image(img: np.ndarray, key, iv, sbox, two_way=True):
    """img: array uint8 HxWxC. Byte ditambah padding nol bila perlu, lalu dipotong lagi."""
    flat = img.astype(np.uint8).tobytes()
    pad = pad_len(len(flat))
    c = encrypt_bytes(flat + b"\0" * pad, key, iv, sbox, two_way)
    return np.frombuffer(c, dtype=np.uint8), pad


def decrypt_image(cipher_flat: np.ndarray, shape, key, iv, sbox, two_way=True):
    p = decrypt_bytes(cipher_flat.astype(np.uint8).tobytes(), key, iv, sbox, two_way)
    n = int(np.prod(shape))
    return np.frombuffer(p[:n], dtype=np.uint8).reshape(shape)


def cipher_to_image(cipher_flat, shape):
    """Ambil n byte pertama untuk ditampilkan sebagai citra (sisanya padding)."""
    n = int(np.prod(shape))
    return cipher_flat[:n].reshape(shape)


# ---------------- Uji citra ----------------
def entropy(channel: np.ndarray) -> float:
    p = np.bincount(channel.ravel(), minlength=256) / channel.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def npcr_uaci(c1: np.ndarray, c2: np.ndarray):
    c1 = c1.astype(np.int64); c2 = c2.astype(np.int64)
    npcr = float((c1 != c2).mean() * 100)
    uaci = float((np.abs(c1 - c2) / 255).mean() * 100)
    return npcr, uaci


def chi_square(channel: np.ndarray) -> float:
    o = np.bincount(channel.ravel(), minlength=256)
    e = channel.size / 256
    return float(((o - e) ** 2 / e).sum())


def correlation(channel: np.ndarray, direction="horizontal", n=3000, seed=0):
    rng = np.random.default_rng(seed)
    h, w = channel.shape
    if direction == "horizontal":
        y = rng.integers(0, h, n); x = rng.integers(0, w - 1, n); a = channel[y, x]; b = channel[y, x + 1]
    elif direction == "vertical":
        y = rng.integers(0, h - 1, n); x = rng.integers(0, w, n); a = channel[y, x]; b = channel[y + 1, x]
    else:
        y = rng.integers(0, h - 1, n); x = rng.integers(0, w - 1, n); a = channel[y, x]; b = channel[y + 1, x + 1]
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])
