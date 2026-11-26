# S-box ML untuk Enkripsi Citra (Streamlit)

Proyek Akhir Kriptografi, Kelompok 1, Universitas Negeri Semarang 2026.

## Isi aplikasi

| Halaman | Isi |
|---|---|
| Beranda | Ringkasan proyek dan metrik S-box yang sedang dipilih |
| Konstruksi S-box (ML) | Membangun dataset dari citra mandrill, melatih Ridge Regression (pembanding: Random Forest), lalu menyusun matriks affine dan S-box |
| Uji S-box | Balance, bijective, NL, SAC, BIC-NL, BIC-SAC, LAP, DAP, fixed point, dan perbandingan dengan AES serta S-box acuan |
| Enkripsi Citra | AES-128 dengan S-box terpilih (CBC dua arah), dekripsi, histogram + χ², entropy, korelasi, NPCR, dan UACI |

S-box bisa dipilih dari panel kiri: S-box ML kelompok, AES standar, tiga S-box acuan, atau S-box unggahan sendiri.

## Menjalankan di komputer sendiri

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud (gratis)

1. Buat repository baru di GitHub, misalnya `sbox-ml-kriptografi`.
2. Unggah semua file di folder ini: `app.py`, `sbox_core.py`, `aes_core.py`, `requirements.txt`, `mandrill.png`, dan folder `.streamlit`.
3. Buka https://share.streamlit.io, login dengan akun GitHub, lalu klik **Create app**.
4. Pilih repository tadi, branch `main`, dan file utama `app.py`, lalu klik **Deploy**.
5. Setelah selesai, salin link aplikasinya (bentuknya `https://nama-app.streamlit.app`) ke Lampiran 2 laporan dan ke Google Form.

## Catatan teknis

- S-box dibangun dengan rumus S(x) = A · x⁻¹ ⊕ 0x63 di GF(2⁸) dengan polinomial 0x1F3.
- AES-128 memakai S-box terpilih di SubBytes dan key expansion, sedangkan MixColumns memakai GF(2⁸) AES standar (0x11B). Implementasi sudah dicek dengan vektor uji FIPS-197 Appendix C.1.
- Kunci bawaan `KRIPTOGRAFI-2026` dan IV `000102…0f` sama dengan yang dipakai di laporan, jadi hasil uji citra sama dengan angka di laporan.
