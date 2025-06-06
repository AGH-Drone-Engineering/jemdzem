import os
import sys
import json
import base64
import argparse
import firebase_admin
import cv2
from firebase_admin import credentials, db, firestore
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import datetime

# Klucz szyfrowania (w produkcji powinien być bezpiecznie przechowywany)
KEY_PASSWORD = b'testowehaslo'
SALT = b'firebase_salt_1234'
backend = default_backend()

def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=backend
    )
    return kdf.derive(password)

def decrypt_file_to_bytes(input_path, password, salt):
    key = derive_key(password, salt)
    with open(input_path, 'rb') as f:
        iv = f.read(16)
        ct = f.read()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data

def decrypt_config(encrypted_file):
    """Odszyfrowuje plik konfiguracyjny."""
    return json.loads(decrypt_file_to_bytes(encrypted_file, KEY_PASSWORD, SALT).decode("utf-8"))

def image_to_base64(image_path):
    """Konwertuje plik obrazu na base64 z kompresją do 64x64 px."""
    # Wczytaj obraz
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    
    # Zmień rozmiar do 64x64 px (mały rozmiar dla Firebase)
    resized_image = cv2.resize(image, (128, 128))
    
    # Kompresuj jako JPEG z umiarkowaną kompresją (jakość = 65)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
    success, encoded_img = cv2.imencode('.jpg', resized_image, encode_param)
    
    if not success:
        raise ValueError("Nie można skompresować obrazu")
    
    # Konwertuj na base64
    return base64.b64encode(encoded_img.tobytes()).decode('utf-8')

def push_point_to_db(point_dict):
    """Dodaje punkt do bazy Firebase i ustawia generate na true."""
    # Pobierz referencję do bazy
    ref = db.reference('/')
    
    # Pobierz aktualne punkty
    data = ref.get()
    points = data.get('points', []) if data else []
    
    # Dodaj nowy punkt
    points.append(point_dict)
    ref.update({'points': points})

def clear_points():
    ref = db.reference('/')
    ref.update({'points': []})

def generate_points():
    ref = db.reference('/')
    ref.update({'generate': True})

DETECTION_LABEL_MAP = {
    "pipe": "pipe",
    "powerpole": "powerpole",
    "barrell": "barrel",
    "palette": "europallet",
    "person": "worker",
    "car": "car"
}

def push_detection_to_firebase(detection, gps_coords, image_path=None):
    """Wysyła detekcję do Firebase za pomocą push_point.py."""
    label = detection["label"]
    firebase_type = DETECTION_LABEL_MAP.get(label, label)  # domyślnie infrastructure
    point_dict = {
        'type': firebase_type,
        'gps_coords': [gps_coords[0], gps_coords[1]],
        'detection_time': datetime.datetime.now().isoformat(),
        'description': detection.get('description', f'Wykryto obiekt typu {label}')
    }
    if image_path:
        try:
            image_b64 = image_to_base64(image_path)
            print(f"Skompresowany obraz base64 length: {len(image_b64)}")
            point_dict['image'] = image_b64
        except Exception as e:
            print(f"Błąd kompresji obrazu: {e}")
            # Kontynuuj bez obrazu
    push_point_to_db(point_dict)

def answer(tekst):
    ref = db.reference('/')
    # Pobierz aktualne punkty
    data = ref.get()
    ref.update({'answer': tekst})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dodaj punkt do bazy Firebase.')
    parser.add_argument('--type', required=True, help='Typ punktu (np. worker, infrastructure, emergency, aruco)')
    parser.add_argument('--lat', type=float, required=True, help='Szerokość geograficzna')
    parser.add_argument('--lon', type=float, required=True, help='Długość geograficzna')
    parser.add_argument('--image', help='Ścieżka do pliku obrazu (opcjonalne)')
    args = parser.parse_args()
    
    point_dict = {
        'type': args.type,
        'gps_coords': [args.lat, args.lon],
        'detection_time': datetime.datetime.now().isoformat()
    }
    
    if args.image:
        point_dict['image'] = image_to_base64(args.image)
    
    push_point_to_db(point_dict)

# Inicjalizacja Firebase przy starcie skryptu
config = decrypt_config(os.path.join(os.path.dirname(__file__), 'firebase_key.json.enc'))
cred = credentials.Certificate(config)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://droniada-2025-default-rtdb.europe-west1.firebasedatabase.app'
})