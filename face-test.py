from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

image = Image.open('test/jisoo2.jpg')

face_location = face_recognition.face_locations(np.array(image))
face_encoding = face_recognition.face_encodings(np.array(image), face_location)

draw = ImageDraw.Draw(image)
face_names = []

threshold = 0.6

for face_encodings, face_locations in zip(face_encoding, face_location):
    
    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
    best_match_index = np.argmin(face_distances)
    
    if face_distances[best_match_index] < threshold:
        name = known_face_names[best_match_index]
    else:
        name = "Unknown"
    
    #วาดรูป
    top, right, bottom, left = face_locations
    draw.rectangle([left, top, right, bottom])
    draw.text((left, top), name)
    face_names.append(name)

print(face_names)
image.show()