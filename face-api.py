from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import io
import os
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import re
from pymongo import MongoClient
from bson.objectid import ObjectId

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

INTERNAL_KEY = os.getenv("INTERNAL_FACE_API_KEY", "dev-internal-key")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://Aadmin:facescan_ab@cluster0.7jxizye.mongodb.net/facescan?retryWrites=true&w=majority&appName=Cluster0")
MONGODB_DB = os.getenv("MONGODB_DB", "facescan")
MONGODB_USERS_COLLECTION = os.getenv("MONGODB_USERS_COLLECTION", "users")

mongo_users = None
if MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        mongo_db = mongo_client[MONGODB_DB]
        mongo_users = mongo_db[MONGODB_USERS_COLLECTION]
        print(f"[Mongo] Connected to {MONGODB_DB}.{MONGODB_USERS_COLLECTION}")
    except Exception as e:
        print(f"[Mongo] Connect FAILED: {e}")
    
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

def is_vec128(v):
    try:
        return isinstance(v, (list, tuple)) and len(v) == 128 and all(isinstance(x, (int, float)) for x in v)
    except:
        return False

def collect_vectors_from_userdoc(doc):
    refs = []
    enc = doc.get("faceEncodings")
    
    if isinstance(enc, list) and len(enc) == 128 and all(isinstance(x, (int, float)) for x in enc):
        refs.append((enc, "enc"))
        
    elif isinstance(enc, list):
        for i, v in enumerate(enc):
            if is_vec128(v):
                refs.append((v, f"enc[{i}]"))
                
    elif isinstance(enc, dict):
        for k in ["front", "left", "right", "up", "down"]:
            v = enc.get(k)
            if is_vec128(v):
                refs.append((v, k))
                
    return refs

def try_objectid(s: str):
    try:
        return ObjectId(str(s))
    except Exception:
        return None

def find_teacher_doc(teacher_id: str):
    """
    พยายามหาเอกสารครูจากหลายคีย์ (_id, teacherId, userId, studentId) ในคอลเลกชัน teachers ก่อน แล้วค่อย users
    """
    idstr = str(teacher_id).strip()
    candidates = []
    oid = try_objectid(idstr)
    if oid:
        candidates.append({"_id": oid})
    candidates += [
        {"teacherId": idstr},
        {"userId": idstr},
        {"studentId": idstr},
    ]
    for coll in [mongo_users]:
        if coll is None:
            continue
        for q in candidates:
            try:
                doc = coll.find_one(q)
                if doc:
                    return doc
            except Exception:
                continue
    return None

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
    request: Request,
    image: UploadFile = File(...),
    studentID: str = Form(...),
    # fullname: str = Form(None),
    threshold: float = Form(...),
):
    # auth ภายใน
    if request.headers.get("x-internal-key") != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    # encode ใบหน้า
    enc = encode_single_face(image)
    if enc is None:
        return {"ok": False, "match": False, "message": "no face detected"}

    # ต้องมีการเชื่อม Mongo
    if mongo_users is None:
        return {"ok": False, "match": False, "message": "model has no DB connection"}

    # ดึงเวกเตอร์อ้างอิงจาก MongoDB
    doc = mongo_users.find_one({"studentId": str(studentID).strip()})
    if not doc:
        return {"ok": False, "match": False, "message": "user not found"}

    refs = collect_vectors_from_userdoc(doc)
    if not refs:
        return {"ok": False, "match": False, "message": "no reference vectors for this user"}

    # เทียบทุกเวกเตอร์
    distances = []
    for vec, label in refs:
        d = float(np.linalg.norm(np.array(vec, dtype=float) - enc))
        distances.append({"label": label, "distance": d})
    distances.sort(key=lambda x: x["distance"])
    best = distances[0]

    thr = float(threshold)
    match = best["distance"] <= thr

    return {
        "ok": True,
        "match": match,
        "distance": best["distance"],
        "threshold": thr,
        "bestRef": best["label"],
        "studentID": str(studentID).strip(),
        "countRefs": len(refs),
    }
    
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

@app.post("/api/scan-teacher")
async def scan_teacher(
    request: Request,
    image: UploadFile = File(...),
    teacherID: str = Form(...),
    threshold: float = Form(...),
):
    # auth ภายใน
    if request.headers.get("x-internal-key") != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    # encode ใบหน้า
    enc = encode_single_face(image)
    if enc is None:
        return {"ok": False, "match": False, "message": "no face detected"}
    
    if not ObjectId.is_valid(teacherID):
        return {"ok": False, "match": False, "message": "invalid teacher _id"}
    
    # DB พร้อมหรือไม่
    if mongo_users is None:
        return {"ok": False, "match": False, "message": "model has no DB connection"}

    # ดึงเวกเตอร์อ้างอิงของครูจาก MongoDB
    doc = mongo_users.find_one({"_id": ObjectId(teacherID)})   # ← ค้นใน users
    if not doc:
        return {"ok": False, "match": False, "message": "teacher not found"}

    refs = collect_vectors_from_userdoc(doc)
    if not refs:
        return {"ok": False, "match": False, "message": "no reference vectors for this teacher"}

    # เทียบทุกเวกเตอร์ -> เอาค่าน้อยสุด
    distances = []
    for vec, label in refs:
        d = float(np.linalg.norm(np.array(vec, dtype=float) - enc))
        distances.append({"label": label, "distance": d})
    distances.sort(key=lambda x: x["distance"])
    best = distances[0]

    thr = float(threshold)
    match = best["distance"] <= thr

    return {
        "ok": True,
        "match": match,
        "distance": best["distance"],
        "threshold": thr,
        "bestRef": best["label"],
        "teacherID": str(teacherID),
        "countRefs": len(refs),
    }