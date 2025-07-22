import cv2
import os

# Menentukan path untuk menyimpan data latih
dataset_path = "facedetection/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Membuka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Membuat objek untuk deteksi wajah (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Menanyakan nama orang yang akan dilatih
name = input("Masukkan nama orang yang akan dilatih: ")

# Membuat folder khusus untuk orang ini
person_path = os.path.join(dataset_path, name)
if not os.path.exists(person_path):
    os.makedirs(person_path)

# Menangkap gambar wajah untuk dilatih
print(f"Mulai mengambil foto wajah untuk {name}. Tekan 'q' untuk berhenti.")
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Mengubah gambar menjadi grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Potong wajah yang terdeteksi
        face = gray[y:y + h, x:x + w]

        # Simpan gambar wajah
        face_filename = os.path.join(person_path, f"{name}_{count}.jpg")
        cv2.imwrite(face_filename, face)
        count += 1

        # Gambarkan kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tampilkan video dengan wajah yang terdeteksi
    cv2.imshow('Capture Faces', frame)

    # Jika sudah mengambil cukup foto, atau tekan 'q' untuk berhenti
    if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Menutup kamera
cv2.destroyAllWindows()  # Menutup jendela OpenCV
print(f"Pengambilan gambar selesai untuk {name}. Total gambar: {count}")
