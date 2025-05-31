from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
try:
    model = load_model("finalModel.keras")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None  # Optional: supaya tidak menyebabkan crash total

# Kelas dan detail
class_info = {
    0: {
        "label": "Nail_psoriasis",
        "arti": "Lorem ipsum dolor sit amet, nail psoriasis is a chronic condition affecting the nails.",
        "saran": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Disarankan untuk konsultasi dengan dokter kulit."
    },
    1: {
        "label": "SJS-TEN",
        "arti": "Lorem ipsum dolor sit amet, SJS-TEN adalah kondisi kulit serius akibat reaksi obat.",
        "saran": "Segera hubungi dokter atau unit gawat darurat untuk penanganan lebih lanjut."
    },
    2: {
        "label": "Vitiligo",
        "arti": "Lorem ipsum dolor sit amet, vitiligo adalah kondisi di mana kulit kehilangan pigmen.",
        "saran": "Penggunaan krim tertentu atau terapi cahaya mungkin disarankan oleh dokter."
    },
    3: {
        "label": "Acne",
        "arti": "Lorem ipsum dolor sit amet, acne adalah kondisi kulit yang umum akibat pori-pori tersumbat.",
        "saran": "Gunakan produk non-komedogenik dan konsultasikan ke dokter kulit untuk penanganan lanjutan."
    },
    4: {
        "label": "Hyperpigmentation",
        "arti": "Lorem ipsum dolor sit amet, hyperpigmentation adalah kondisi kulit dengan warna lebih gelap.",
        "saran": "Gunakan tabir surya dan krim pencerah, serta konsultasikan jika tidak membaik."
    }
}

# Preprocessing image
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        result = {
            "label": class_info[predicted_class]["label"],
            "arti": class_info[predicted_class]["arti"],
            "saran": class_info[predicted_class]["saran"],
            "confidence": round(confidence, 3)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Skin Disease Detection API with details is running."

if __name__ == "__main__":
    app.run(debug=True)
