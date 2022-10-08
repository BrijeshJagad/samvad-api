from flask import Flask, request, jsonify, render_template
import requests, json
from funs import create_model, predict

model = create_model('model.onnx')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST', 'GET'])
def translate():
    video_path = request.form.get('video') #video path
#     video_path = "VN20221006_123507.mp4"
    # result = {'pictures': pictures}

#     res = json.loads(requests.get(video_path).content.decode("utf-8"))
#     video_path += "?alt=media&token=" + res["downloadTokens"]

    text = predict(model, video_path)

    translation = {'Arm' : 'હાથ', 'Fly' : 'ઉડી', 'Thank you' : 'આભાર', 'Window' : 'બારી', 'Dog' : 'કૂતરો', 'Couldn\'t recognize' : 'ઓળખી ન શક્યા'}
    result = {
        'english': text,
        'gujarati' : translation[text]
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run()
