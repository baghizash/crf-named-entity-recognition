# Named Entity Recognition Alamat dengan CRF

Mengenali field alamat (address, city, state, zip) pada tiap token memakai
Conditional Random Field. CRF memodelkan urutan label sehingga transisi antar
label ikut dipelajari, cocok untuk data alamat yang polanya berurutan.

Proyek ini adalah Module 2 dari kursus Applied Information Extraction in
Python (University of Michigan).

## Data

Folder `data` berisi `list_of_real_usa_addresses.json`: 234 alamat nyata di
AS, masing masing dengan field address, city, state, dan zip. Setelah
tokenisasi per kata, terkumpul 1550 token: address 816, city 266, state 234,
zip 234.

## Yang dikerjakan

1.  Tokenisasi tiap field menjadi kata dan pelabelan per token. Fitur tiap
    token: kata itu sendiri, is_digit, panjang kata, is_upper.
2.  Latih CRF (algoritma lbfgs, c1 0.1, c2 0.1, 100 iterasi) pada 80 persen
    data. Akurasi flat pada 20 persen data uji = 0.9685. Lihat
    `charts/02_f1_per_label.png`.
3.  Skor F1 per label: state 1.000, zip 1.000, address 0.970, city 0.907.
    Label city paling sulit karena beberapa nama kota terdiri dari banyak
    kata. Lihat `charts/01_distribusi_label.png`.
4.  Visualisasi bobot transisi dan bobot fitur state yang dipelajari CRF,
    misalnya transisi kuat dari address ke address dan city ke state. Lihat
    `charts/03_bobot_transisi.png` dan `charts/04_bobot_fitur.png`.

## Cara menjalankan

```bash
python analyze_crf.py
```

Script ini memuat data, melatih CRF, menyimpan chart ke folder `charts`,
dan menulis ringkasan angka ke `findings.json`. Perlu paket
`sklearn_crfsuite`.

## Tools

Python, sklearn_crfsuite, scikit learn, pandas, matplotlib.
