from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import whisper
import os
from moviepy.editor import AudioFileClip
import tempfile

app = Flask(__name__)
CORS(app)

# Load the Whisper model
model = whisper.load_model("base")

def convert_mp4_to_wav(mp4_file_path, wav_file_path):
    with AudioFileClip(mp4_file_path) as audio:
        audio.write_audiofile(wav_file_path, codec='pcm_s16le')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 업로드된 파일을 임시 파일로 저장
        temp_mp4_path = os.path.join(temp_dir, file.filename)
        file.save(temp_mp4_path)

        # mp4 파일을 wav로 변환
        temp_wav_path = os.path.join(temp_dir, 'converted_file.wav')
        convert_mp4_to_wav(temp_mp4_path, temp_wav_path)

        # librosa를 사용하여 wav 파일 로드
        audio, sr = librosa.load(temp_wav_path, sr=16000)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

    return jsonify({"transcription": result.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
