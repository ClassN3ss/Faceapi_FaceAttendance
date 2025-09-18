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

# 🔴 Endpoint ใหม่: ตรวจสอบภาพ 5 มุม และบันทึกภาพ + vector ถ้า verified
@app.post("/api/verify-face")
async def verify_face(
    fullname: str = Form(...),
    studentID: str = Form(...),
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    up: UploadFile = File(...),
    down: UploadFile = File(...),
):
    def image_to_encoding(upload: UploadFile):
        if not upload:
            return None
        upload.file.seek(0)
        image = Image.open(io.BytesIO(upload.file.read()))
        np_image = np.array(image)
        face_locations = face_recognition.face_locations(np_image)
        face_encodings = face_recognition.face_encodings(np_image, face_locations)
        return face_encodings[0] if face_encodings else None

    # 1) encode ทั้ง 5 มุม (บังคับทุกมุมต้องมีและต้องเจอหน้า)
    uploads = {
        "front": front,
        "left": left,
        "right": right,
        "up": up,
        "down": down,
    }

    encodings = {}
    for key, file in uploads.items():
        enc = image_to_encoding(file)
        if enc is None:
            return {"verified": False, "message": f"Face not detected in {key} image"}
        encodings[key] = enc.tolist()

    # 2) เปรียบเทียบ extra (left, right, up, down) กับ front
    threshold = 0.5
    enc_front = np.array(encodings["front"], dtype=float)
    match_count = 0

    for key in ["left", "right", "up", "down"]:
        enc_extra = np.array(encodings[key], dtype=float)
        matches = face_recognition.compare_faces([enc_front], enc_extra, tolerance=threshold)
        if bool(matches[0]):
            match_count += 1

    verified = match_count >= 2

    return {
        "verified": verified,
        "matchCount": match_count,
        "threshold": threshold,
        "encodings": encodings if verified else None,
    }

@app.post("/api/scan-face")
async def scan_face(
    request: Request,
    image: UploadFile = File(...),
    studentID: str = Form(...),
    threshold: float = Form(...),
):
    if request.headers.get("x-internal-key") != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    enc = encode_single_face(image)
    if enc is None:
        return {"ok": False, "match": False, "message": "no face detected"}

    if mongo_users is None:
        return {"ok": False, "match": False, "message": "model has no DB connection"}

    doc = mongo_users.find_one({"studentId": str(studentID).strip()})
    if not doc:
        return {"ok": False, "match": False, "message": "user not found"}

    refs = collect_vectors_from_userdoc(doc)
    if not refs:
        return {"ok": False, "match": False, "message": "no reference vectors for this user"}

    distances = []
    for vec, label in refs:
        distance = face_recognition.face_distance([np.array(vec, dtype=float)], enc)[0]
        matches = face_recognition.compare_faces([np.array(vec, dtype=float)], enc, tolerance=float(threshold))
        distances.append({"label": label, "distance": float(distance), "match": bool(matches[0])})
    distances.sort(key=lambda x: x["distance"])
    best = distances[0]

    return {
        "ok": True,
        "match": best["match"],
        "distance": best["distance"],
        "threshold": float(threshold),
        "bestRef": best["label"],
        "studentID": str(studentID).strip(),
        "countRefs": len(refs),
    }

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

        return {
            "ok": True,
            "descriptor": desc,
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
    if request.headers.get("x-internal-key") != INTERNAL_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    enc = encode_single_face(image)
    if enc is None:
        return {"ok": False, "match": False, "message": "no face detected"}
    
    if not ObjectId.is_valid(teacherID):
        return {"ok": False, "match": False, "message": "invalid teacher _id"}
    
    if mongo_users is None:
        return {"ok": False, "match": False, "message": "model has no DB connection"}

    doc = mongo_users.find_one({"_id": ObjectId(teacherID)})
    if not doc:
        return {"ok": False, "match": False, "message": "teacher not found"}

    refs = collect_vectors_from_userdoc(doc)
    if not refs:
        return {"ok": False, "match": False, "message": "no reference vectors for this teacher"}

    distances = []
    for vec, label in refs:
        distance = face_recognition.face_distance([np.array(vec, dtype=float)], enc)[0]
        matches = face_recognition.compare_faces([np.array(vec, dtype=float)], enc, tolerance=float(threshold))
        distances.append({"label": label, "distance": float(distance), "match": bool(matches[0])})
    distances.sort(key=lambda x: x["distance"])
    best = distances[0]

    return {
        "ok": True,
        "match": best["match"],
        "distance": best["distance"],
        "threshold": float(threshold),
        "bestRef": best["label"],
        "teacherID": str(teacherID),
        "countRefs": len(refs),
    }