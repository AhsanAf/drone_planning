20/4/2026
penambahan file drone_supervisor_pt2
fungsi : drone yang nantinya sudah ada pid dan fisika diharapkan bisa terbang dengan normal

edit bagian world info
WorldInfo {
  basicTimeStep 8
  defaultDamping Damping {
    linear 0.5
    angular 0.5
  }
}
hasil : Masih error di bagian pid

tambahkan pip PID
pip install simple_pid
hasil : command berjalan tetapi gui tidak menerima traceback dari drone_supervisor

21/4/2026
update : propeler drone sudah bisa berputar tetapi masih stand by belum ada take off (edit code drone_supervisor_pt2)
update : dicoba dengan program bawaan webots mavic2pro dan bisa dikontrol dengan normal bisa naik turun maju mundur
tidak seperti sebelumnya pada saat menggunakan device mac (masalah belum ditemukan)
update : drone sudah bisa take off namun berputar tidak jelas dan terpental
indikasi : kemungkinan setting PID yang kurang cocok
