# # 1. CEK FACE RECOGNITION UNTUK GAMBAR RGB

# import face_recognition
# import cv2

# # Memuat gambar
# image = cv2.imread('E:/Zharfan File/Profil for game/46412.jpg') #GANTI DENGAN PATH FOTOMU
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Mendeteksi wajah
# face_locations = face_recognition.face_locations(rgb_image)

# # Menandai wajah
# for (top, right, bottom, left) in face_locations:
#     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# # Tampilkan hasil
# cv2.imshow('Deteksi Wajah', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 2. CEK FACE RECOGNITION UNTUK GAMBAR RGB (DIBUAT GRAYSCALE DULU)

# import face_recognition
# import cv2

# # Memuat gambar grayscale
# image_gray = cv2.imread('E:/Zharfan File/Profil for game/46412.jpg', cv2.IMREAD_GRAYSCALE) #GANTI DENGAN PATH FOTOMU

# # Konversi gambar grayscale ke RGB
# image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

# # Mendeteksi wajah
# face_locations = face_recognition.face_locations(image_rgb)

# # Menandai wajah
# image_with_faces = image_gray.copy()  # Copy gambar grayscale untuk ditampilkan
# for (top, right, bottom, left) in face_locations:
#     cv2.rectangle(image_with_faces, (left, top), (right, bottom), (0, 255, 0), 2)

# # Tampilkan hasil deteksi wajah
# cv2.imshow('Deteksi Wajah Grayscale', image_with_faces)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
