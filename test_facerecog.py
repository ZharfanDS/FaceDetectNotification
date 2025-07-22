import os
import cv2
import face_recognition
import telegram
import asyncio
import time

# Konfigurasi Telegram Bot
BOT_TOKEN = ""  # Ganti dengan token bot Anda
CHAT_ID = ""  # Ganti dengan Chat ID Anda
bot = telegram.Bot(token=BOT_TOKEN)

# Menentukan path dataset secara dinamis
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset")

# Membaca dataset wajah dan encoding
known_face_encodings = []
known_face_names = []

# Iterasi untuk setiap folder dalam dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    # Iterasi untuk setiap gambar di dalam folder orang
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Memuat gambar wajah
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah dan encoding wajah
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Jika wajah terdeteksi, tambahkan encoding wajah ke list
        for face_encoding in face_encodings:
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)

# Membuka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Mengatur resolusi frame untuk mempercepat pemrosesan
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Menyimpan waktu pengiriman notifikasi terakhir
last_face_detection_time = 0
notification_interval = 5  # Interval notifikasi dalam detik

async def send_notification(name, image_path):
    """Fungsi untuk mengirimkan notifikasi secara asynchronous"""
    await bot.send_message(chat_id=CHAT_ID, text=f"Wajah terdeteksi: {name}")
    with open(image_path, 'rb') as photo:
        await bot.send_photo(chat_id=CHAT_ID, photo=photo)
    os.remove(image_path)  # Hapus gambar setelah dikirim
    print(f"Gambar {image_path} telah dihapus setelah dikirim.")

async def process_video_frames():
    """Fungsi untuk memproses frame video secara asynchronous"""
    global last_face_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # Mengubah gambar menjadi RGB untuk face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah pada frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Membandingkan encoding wajah dengan dataset
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"  # Nama default jika tidak ditemukan kecocokan

            # Jika ada kecocokan, ambil nama orang tersebut
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Gambarkan kotak dan nama di sekitar wajah
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mengirim notifikasi jika wajah terdeteksi dan interval waktu sudah cukup
            current_time = time.time()
            if current_time - last_face_detection_time > notification_interval:
                print(f"Wajah terdeteksi! Mengirim notifikasi... Nama: {name}")
                image_path = f"screenshot_{name}.png"
                cv2.imwrite(image_path, frame)  # Simpan frame sebagai gambar
                await send_notification(name, image_path)  # Kirim notifikasi
                last_face_detection_time = current_time

        # Tampilkan video dengan wajah yang terdeteksi dan nama
        cv2.imshow('Face Recognition', frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Menutup kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

# Menjalankan deteksi wajah menggunakan asyncio
async def main():
    await process_video_frames()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Deteksi dihentikan.")
    cap.release()
    cv2.destroyAllWindows()
