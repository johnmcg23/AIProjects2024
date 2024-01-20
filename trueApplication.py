from flask import Flask, jsonify
import threading

from Service.RecogniseFaceService import main as recognise_face_main
from Service.RegisterFaceService import main as register_face_main

app = Flask(__name__)

from concurrent.futures import ThreadPoolExecutor

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)


@app.route('/recognise/faceid', methods=['GET', 'POST'])
def runFacialRecognitionLogin():
    print("Gets in recognise Controller")
    # Start the face recognition in a new thread
    result = threading.Thread(target=recognise_face_main).start()
    print(f"the result of your face being in the repo is: {result}")
    return jsonify(True)


@app.route('/register/faceid', methods=['GET', 'POST'])
def runFacialRecognitionSignUp():
    print("Gets in register Controller")
    # Start the face recognition in a new thread
    threading.Thread(target=register_face_main).start()

    return jsonify(True)


if __name__ == '__main__':
    app.run(host='localhost', port=3002, debug=True)
