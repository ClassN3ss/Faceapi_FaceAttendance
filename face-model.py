from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

known_faces = [
    ('Lisa', 'lisa.jpg'),
    ('Jennie', 'jennie.jpg'),
    ('Rose', 'rose.jpg'),
    ('Jisoo', 'jisoo.jpg')
]

known_face_names = []
known_face_encodings = []

for face in known_faces:
    try:
        # name = face[0]
        # filename = face[1]
        
        face_image = face_recognition.load_image_file(face[1])
        face_encoding = face_recognition.face_encodings(face_image)
        
        if face_encoding:
            known_face_names.append(face[0])
            known_face_encodings.append(face_encoding[0])
        else:
            print(f"❗ ไม่พบใบหน้าในรูป {face[1]} (ของ {face[0]})")
    
    except FileNotFoundError:
        print(f"⚠️ ไม่พบไฟล์: {face[1]}")

pickle.dump((known_face_names, known_face_encodings), open('faces.p', 'wb'))