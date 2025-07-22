import cv2
import os
import numpy as np
import telegram
import asyncio

# Konfigurasi Telegram Bot
BOT_TOKEN = ""  # Ganti dengan token bot Anda
CHAT_ID = ""  # Ganti dengan Chat ID Anda
bot = telegram.Bot(token=BOT_TOKEN)

# Memuat model Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(r'D:/Program (Software)/Visual Studio Code CODE/belajar-Python/facedetection/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml') # Ganti ke file haarcascade_frontalface_default.xml di folder anda!

# Menginisialisasi pengenalan wajah
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Menentukan path dataset secara dinamis
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset") # dataset or try facedetection/dataset

people = os.listdir(dataset_path)  # Daftar orang yang dilatih
faces, labels = [], []

# Mengambil data latih dari dataset
for person_name in people:
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(gray)
            labels.append(people.index(person_name))

# Melatih model pengenalan wajah
recognizer.train(faces, np.array(labels))

# Membuka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Variabel untuk mendeteksi apakah notifikasi sudah dikirim dalam satu siklus deteksi
last_face_detection_time = 0
notification_interval = 5  # Waktu tunggu dalam detik untuk mengirimkan notifikasi lagi

# Fungsi untuk mengirimkan notifikasi secara asynchronous
async def send_notification(name, image_path):
    """Fungsi untuk mengirimkan notifikasi dan gambar secara asynchronous"""
    # Kirim pesan teks
    await bot.send_message(chat_id=CHAT_ID, text=f"Seseorang yang dikenali dengan nama {name} terdeteksi di depan kamera!")
    
    # Kirim gambar
    with open(image_path, 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo)
    
    # Hapus gambar setelah dikirim
    os.remove(image_path)
    print(f"Gambar {image_path} telah dihapus setelah dikirim.")

# Menangani identifikasi wajah
async def identify_faces():
    global last_face_detection_time  # Menambahkan global untuk mengakses variabel ini dalam fungsi
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Gagal membaca frame.")
            break
        
        # Konversi frame ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Gambar kotak di sekitar wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Potong wajah dari frame
            face_region = gray[y:y + h, x:x + w]

            # Kenali wajah
            label, confidence = recognizer.predict(face_region)

            # Ambil nama orang berdasarkan label yang terdeteksi
            name = people[label] if confidence < 100 else "Tidak Dikenali"

            # Tampilkan nama orang yang terdeteksi di layar
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Cek apakah wajah terdeteksi dan kirim notifikasi
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_face_detection_time > notification_interval:
                print(f"Wajah terdeteksi! {name}")
                
                # Simpan screenshot dan kirim gambar ke Telegram
                image_path = f"screenshot_{name}.png"
                cv2.imwrite(image_path, frame)
                await send_notification(name, image_path)
                last_face_detection_time = current_time  # Update waktu terakhir notifikasi dikirim

        # Tampilkan video dengan deteksi wajah
        cv2.imshow('Deteksi Wajah', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Menutup kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV
    print("Kamera dan jendela ditutup.")

# Menjalankan event loop utama
try:
    asyncio.run(identify_faces())  # Menjalankan event loop
except RuntimeError as e:
    print(f"Runtime error occurred: {e}")
    cap.release()  # Pastikan kamera ditutup jika terjadi error
    cv2.destroyAllWindows()  # Pastikan jendela ditutup jika terjadi error
