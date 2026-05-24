import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

BASE_URL = "http://192.168.1.36:8000"

def imagen_base64(siglas, color_fondo):
    img = Image.new("RGB", (200, 200), color=color_fondo)
    draw = ImageDraw.Draw(img)
    # Texto centrado
    draw.text((100, 100), siglas, fill="white", anchor="mm")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

partidos = [
    {"nombre": "Partido Democrático Nacional", "siglas": "PDN", "color": (180, 30, 30)},
    {"nombre": "Alianza por el Progreso",      "siglas": "APP", "color": (30, 30, 180)},
    {"nombre": "Frente Popular Unido",          "siglas": "FPU", "color": (30, 150, 30)},
]

for p in partidos:
    r = requests.post(f"{BASE_URL}/admin/partidos", json={
        "nombre":     p["nombre"],
        "siglas":     p["siglas"],
        "foto_base64": imagen_base64(p["siglas"], p["color"])
    })
    print(f"{p['siglas']}: {r.status_code} - {r.json()}")