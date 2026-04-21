
## PANDUAN INSTALASI & CARA MENJALANKAN PROGRAM

Ikuti langkah-langkah di bawah ini secara berurutan untuk menyiapkan 
program di komputer.

-----------------
## 1. PERSIAPAN AWAL

Pastikan komputer sudah memiliki Python 3.14.
- Download di: https://www.python.org/downloads/

------------------------------------
## 2. CARA INSTALASI VIRTUAL ENVIRONMENTS

Buka folder program ini, lalu buka Terminal (Mac/Linux) atau 
Command Prompt/PowerShell (Windows). Ketik perintah berikut:

A. Buat Ruang Kerja Khusus (Virtual Environment)
   Ketik perintah ini dan tekan Enter:
   
   Windows:
   ``` bash 
   python -m venv .venv
```
   
   Mac/Linux:
   ``` bash
python3 -m venv .venv
   ```

B. Aktifkan Ruang Kerja
   Setelah dibuat, kita harus "masuk" ke dalamnya:

   Windows (Command Prompt):
   ``` bash
   .venv\Scripts\activate
```

   Mac/Linux:
   ``` bash
   source .venv/bin/activate
```

   TANDA BERHASIL: Muncul tulisan (.venv) di baris ketikan Anda.

C. Instal Bahan-Bahan (Library)
   Pastikan (.venv) sudah aktif, lalu ketik:
   
   ``` bash
   pip install -r requirements.txt
```
   
------------------------------------
## 3. CARA MENJALANKAN PROGRAM

Setelah instalasi selesai, jalankan program dengan perintah:

``` bash
python drone_gui.py
```

---------------------------

## 4. CARA BERHENTI/KELUAR

Jika sudah selesai, cukup ketik:

```bash
deactivate
```

-----------------------
