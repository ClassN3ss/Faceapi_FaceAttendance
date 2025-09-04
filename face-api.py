from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import io
import os
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import requests
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://be-attendance-4cec7c12e4bd.herokuapp.com/")
INTERNAL_KEY = os.getenv("INTERNAL_FACE_API_KEY", "dev-internal-key")
VERIFY_VECTOR_ENDPOINT = f"{BACKEND_BASE_URL}/api/face/verify-vector-by-id"

# โหลดฐานข้อมูลใบหน้า
try:
    known_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))
except:
    known_face_names, known_face_encodings = [], []
    
def encode_single_face(upload: UploadFile):
    """
    อ่านรูปจาก UploadFile -> หาใบหน้า -> คืนเวกเตอร์ใบหน้าใบแรก (numpy array 128D)
    ถ้าไม่เจอใบหน้า -> คืน None
    """
    upload.file.seek(0)
    img = Image.open(io.BytesIO(upload.file.read()))
    np_img = np.array(img)
    locations = face_recognition.face_locations(np_img)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(np_img, locations)
    return encodings[0] if encodings else None

# 🔵 Endpoint เดิม: ตรวจสอบชื่อจากภาพเดียว
@app.post("/faces_recognition/")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()
    image = Image.open(io.BytesIO(data))
    
    face_location = face_recognition.face_locations(np.array(image))
    face_encoding = face_recognition.face_encodings(np.array(image), face_location)
    
    draw = ImageDraw.Draw(image)
    face_names = []

    if face_encoding:
        for face_encodings, face_locations in zip(face_encoding, face_location):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
            best_match_index = np.argmin(face_distances)
            
            threshold = 0.5
            if face_distances[best_match_index] < threshold:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
            
            top, right, bottom, left = face_locations
            draw.rectangle([left, top, right, bottom])
            draw.text((left, top), name)
            face_names.append(name)
    else:
        print("❗ ไม่พบใบหน้าในภาพ")
    
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format='PNG')
    image_byte_arr = image_byte_arr.getvalue()
    
    return StreamingResponse(io.BytesIO(image_byte_arr), media_type='image/png')

# 🔴 Endpoint ใหม่: ตรวจสอบภาพ 5 มุม และบันทึกภาพ + vector ถ้า verified
@app.post("/api/verify-face")
async def verify_face(
    fullname: str = Form(...),
    studentID: str = Form(...),
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    up: UploadFile = File(...),
    down: UploadFile = File(...)
):
    def image_to_encoding(upload: UploadFile):
        upload.file.seek(0)
        image = Image.open(io.BytesIO(upload.file.read()))
        np_image = np.array(image)
        face_locations = face_recognition.face_locations(np_image)
        face_encodings = face_recognition.face_encodings(np_image, face_locations)
        return (image, face_encodings[0]) if face_encodings else (image, None)

    # สร้าง encoding พร้อมเก็บตัวภาพไว้ใช้เซฟไฟล์
    images = {}
    encodings = {}

    for key, upload in [("front", front), ("left", left), ("right", right), ("up", up), ("down", down)]:
        img, enc = image_to_encoding(upload)
        images[key] = img
        encodings[key] = enc

    # ต้องเจอหน้าในทุกมุม
    if any(enc is None for enc in encodings.values()):
        return {"verified": False, "message": "Face not found in all images"}

    # เทียบหน้าตรงกับอีก 4 มุม
    front_encoding = encodings["front"]
    threshold = 0.5
    match_count = 0
    for direction in ["left", "right", "up", "down"]:
        distance = np.linalg.norm(encodings[direction] - front_encoding)
        if distance < threshold:
            match_count += 1

    verified = (match_count >= 3)  # 3 ใน 4 = true, อื่นๆ false (รวมถึง 1/4 และ 2/4)
    
    def safe_id(s: str) -> str:
        # อนุญาตตัวเลข/ตัวอักษร/_- (คงเครื่องหมาย - ไว้สำหรับรูปแบบ 64-040603-4567-0)
        return re.sub(r"[^0-9A-Za-z_-]", "", str(s or "")).strip()
    # เซฟรูปเสมอหรือเฉพาะตอน verified?
    # ตามสเปก: เซฟรูปเมื่อ verified เท่านั้น
    saved_paths = {}
    if verified:
        sid = safe_id(studentID)            # ← ใช้ studentID เป็นชื่อโฟลเดอร์
        folder_path = os.path.join("user", sid)
        os.makedirs(folder_path, exist_ok=True)

        for key in ["front", "left", "right", "up", "down"]:
            filename = f"{fullname}_{key}.jpg"
            out_path = os.path.join(folder_path, filename)
            print("💾 Saving image to:", out_path)
            images[key].save(out_path)
            saved_paths[key] = out_path

    # ส่งเวกเตอร์กลับให้ backend เก็บลง MongoDB
    # แปลง numpy -> list
    enc_as_list = {k: enc.tolist() for k, enc in encodings.items()}

    return {
        "verified": verified,
        "matchCount": match_count,
        "threshold": threshold,
        "encodings": enc_as_list if verified else None,
        "savedPaths": saved_paths if verified else None
    }

@app.post("/api/scan-face")
async def scan_face(
    fullname: str = Form(...),
    studentID: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        enc = encode_single_face(image)
        if enc is None:
            return {"ok": False, "match": False, "message": "❌ ไม่พบใบหน้าในภาพ"}

        payload = {
            "studentID": str(studentID).strip(),
            "faceVector": enc.tolist(),
        }

        resp = requests.post(
            VERIFY_VECTOR_ENDPOINT,
            json=payload,
            headers={"x-internal-key": INTERNAL_KEY, "Content-Type": "application/json"},
            timeout=10,
        )

        # เผื่อ backend ตอบไม่ใช่ 200
        if resp.status_code != 200:
            try:
                data = resp.json()
            except Exception:
                data = {"message": resp.text}
            return {"ok": False, "match": False, "message": data.get("message", "ตรวจสอบไม่สำเร็จ")}

        data = resp.json()
        if not data.get("ok"):
            return {"ok": False, "match": False, "message": data.get("message", "ตรวจสอบไม่สำเร็จ")}

        return {
            "ok": True,
            "match": bool(data.get("match")),
            "distance": data.get("distance"),
            "threshold": data.get("threshold"),
            "studentId": data.get("studentID"),
            "fullName": fullname,
        }

    except Exception as e:
        return {"ok": False, "match": False, "message": f"❌ Error: {str(e)}"}
    
def sanitize(name: str) -> str:
    # ตัดอักขระที่ใช้ตั้งชื่อโฟลเดอร์ไม่ได้
    return "".join(c for c in name.strip() if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEACHER_DIR = os.path.join(BASE_DIR, "teacher")
os.makedirs(TEACHER_DIR, exist_ok=True)

@app.post("/api/teacher-saveface")
async def encode_and_save(image: UploadFile = File(...), fullname: str = Form(...)):
    try:
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)

        boxes = face_recognition.face_locations(arr, model="hog")
        if not boxes:
            return {"ok": False, "message": "no face detected"}
        encs = face_recognition.face_encodings(arr, known_face_locations=[boxes[0]])
        if not encs:
            return {"ok": False, "message": "cannot encode face"}

        desc = encs[0].tolist()

        # ✅ ใช้ fullname โดยตรงเป็นชื่อโฟลเดอร์
        folder = fullname.strip() or "unknown"
        folder_path = os.path.join(TEACHER_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)

        # ใช้ชื่อไฟล์เป็น fullname เช่นกัน
        filename = f"{folder}.jpg"
        save_path = os.path.join(folder_path, filename)
        img.save(save_path, format="JPEG")

        rel_path = os.path.relpath(save_path, BASE_DIR).replace("\\", "/")

        return {
            "ok": True,
            "descriptor": desc,
            "imagePath": rel_path,
            "personKey": folder,   # คราวนี้ personKey = "ผู้ช่วยศาสตราจารย์ มะนาว ส้มโอ"
            "fullname": fullname.strip()
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/teacher-scan")
async def teacher_encode(image: UploadFile = File(...), fullname: str = Form(...)):
    try:
        data = await image.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        boxes = face_recognition.face_locations(arr, model="hog")
        if not boxes:
            return {"ok": False, "message": "no face detected"}
        encs = face_recognition.face_encodings(arr, known_face_locations=[boxes[0]])
        if not encs:
            return {"ok": False, "message": "cannot encode face"}
        return {"ok": True, "descriptor": encs[0].tolist(), "fullName": fullname}
    except Exception as e:
        return {"ok": False, "message": str(e)}