# app.py (Tanpa Emoji)
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)
MODEL_FILENAME = 'sleep_quality_model_v3.pkl'

# Fungsi untuk memetakan Skala 1-10 ke Nilai Menit/Hari
def map_activity_scale_to_minutes(scale):
    """Memetakan skala 1-10 ke menit aktivitas fisik (10x skala)."""
    return int(scale * 10)

# --- 1. Persiapan dan Pelatihan Model ---
# ... (Kode train_model tidak berubah) ...
def train_model(data_path='Sleep_health_and_lifestyle_dataset.csv'):
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)
        features = ['Age', 'Sleep Duration', 'Stress Level', 'Physical Activity Level']
        target = 'Quality of Sleep'
        if not all(col in df.columns for col in features + [target]):
             raise ValueError("Kolom yang dibutuhkan tidak lengkap di dataset.")
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILENAME)
        print("Model berhasil dilatih dan disimpan.")
        return model
    except FileNotFoundError:
        print(f"Error: File dataset '{data_path}' tidak ditemukan. Mohon letakkan di folder yang sama.")
        return None
    except Exception as e:
        print(f"Terjadi error saat melatih model: {e}")
        return None

try:
    model = joblib.load(MODEL_FILENAME)
    print("Model yang sudah ada dimuat.")
except (FileNotFoundError, Exception):
    print("Model tidak ditemukan atau gagal dimuat, melatih model baru...")
    model = train_model()

# --- 2. Fungsi Pembantu (Tanpa Emoji) ---
def interpret_prediction(score):
    """Memberikan interpretasi kualitatif berdasarkan skor 1-10."""
    if score >= 8.5:
        return "Tidur Sangat Nyenyak (Excellent)"
    elif score >= 7.0:
        return "Tidur Nyenyak (Good)"
    elif score >= 5.5:
        return "Kualitas Tidur Sedang (Average)"
    elif score >= 4.0:
        return "Kualitas Tidur Kurang (Poor)"
    else:
        return "Kualitas Tidur Sangat Buruk (Very Poor)"


# --- 3. Rute Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model prediksi belum siap. Mohon pastikan file dataset ada.'}), 500

    try:
        data = request.form
        
        usia = int(data.get('usia'))
        durasi_tidur = float(data.get('durasi_tidur'))
        tingkat_stress = int(data.get('tingkat_stress'))
        aktivitas_skala = int(data.get('aktivitas_fisik_skala'))
        
        aktivitas_fisik_menit = map_activity_scale_to_minutes(aktivitas_skala)
        
        input_data = np.array([[usia, durasi_tidur, tingkat_stress, aktivitas_fisik_menit]])
        
        prediction = model.predict(input_data)[0]
        
        predicted_score = max(1.0, min(10.0, prediction))
        kualitas_tidur_text = interpret_prediction(predicted_score)
        
        return jsonify({
            'skor_prediksi': f'{predicted_score:.2f} / 10',
            'kualitas_tidur': kualitas_tidur_text,
            'detail_input': f"Usia: {usia}, Durasi Tidur: {durasi_tidur} jam, Stress: {tingkat_stress}, Aktivitas Fisik: Skala {aktivitas_skala} (~{aktivitas_fisik_menit} min/hari)"
        })

    except (ValueError, TypeError):
        return jsonify({'error': 'Pastikan semua input diisi dengan angka yang valid dan tidak ada kolom yang kosong.'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan server: {e}'}), 500

if __name__ == '__main__':
    if model:
        print("Aplikasi Flask berjalan di http://127.0.0.1:5000/")
        app.run(debug=True, use_reloader=False) 
    else:
        print("Gagal menjalankan aplikasi karena model tidak dapat disiapkan.")