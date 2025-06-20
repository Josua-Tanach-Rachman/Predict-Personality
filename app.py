from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load best trained model
model = joblib.load('model/best_model.pkl')

# Route untuk halaman utama (form input user)
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk menangani prediksi saat form disubmit
@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data dari form dan mengubahnya ke tipe data yang sesuai
    try:
        time_alone = int(request.form['Time_spent_Alone'])
        stage_fear = 1 if request.form['Stage_fear'].lower() == 'yes' else 0
        event_attend = int(request.form['Social_event_attendance'])
        going_outside = int(request.form['Going_outside'])
        drained = 1 if request.form['Drained_after_socializing'].lower() == 'yes' else 0
        friends_size = int(request.form['Friends_circle_size'])
        post_freq = int(request.form['Post_frequency'])

        # Menggabungkan semua fitur menjadi satu array numpy
        features = np.array([[time_alone, stage_fear, event_attend, going_outside, drained, friends_size, post_freq]])

        # Melakukan prediksi menggunakan model
        prediction = model.predict(features)[0]

        # Menginterpretasi hasil prediksi
        personality = 'Introvert' if prediction == 1 else 'Extrovert'

        # Menampilkan hasil prediksi ke halaman HTML
        return render_template('index.html', prediction_text=f'Predicted Personality: {personality}')
    
    except Exception as e:
        # Jika terjadi error (misalnya input tidak valid), tampilkan pesan error
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
