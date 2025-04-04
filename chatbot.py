from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Konfigurasi database SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model database untuk pertanyaan dan jawaban chatbot
class ChatbotResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255), unique=True, nullable=False)
    answer = db.Column(db.String(255), nullable=False)

# Buat database dan tabel jika belum ada
with app.app_context():
    db.create_all()

# Load model DialoGPT
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fungsi untuk mendapatkan respons dari DialoGPT
def chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower()
    
    # Cek di database dulu
    response = ChatbotResponse.query.filter_by(question=user_message).first()
    if response:
        return jsonify({"response": response.answer})
    
    # Jika tidak ada di database, gunakan DialoGPT
    ai_response = chatbot_response(user_message)
    return jsonify({"response": ai_response})

@app.route("/learn", methods=["POST"])
def learn():
    data = request.json
    question = data.get("question", "").lower()
    answer = data.get("answer", "")
    
    if question and answer:
        existing_response = ChatbotResponse.query.filter_by(question=question).first()
        if not existing_response:
            new_response = ChatbotResponse(question=question, answer=answer)
            db.session.add(new_response)
            db.session.commit()
            return jsonify({"response": "Terima kasih! Saya telah belajar jawaban baru."})
        else:
            return jsonify({"response": "Jawaban sudah ada di database."})
    return jsonify({"response": "Gagal menyimpan jawaban, pastikan input benar."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)