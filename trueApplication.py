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
    print("Gets in method in python")

    # Start the face recognition in a new thread and get a Future object
    future = executor.submit(recognise_face_main)

    # Get the result from the Future
    result = future.result()

    # Return the result
    if result:
        print("Face is in storage")
        return jsonify(True)
    else:
        print("Face is NOT in storage")
        return jsonify(False)


@app.route('/register/faceid', methods=['GET', 'POST'])
def runFacialRecognitionSignUp():
    print("Getting into sign up backend")
    # Start the face recognition in a new thread
    threading.Thread(target=register_face_main).start()

    return jsonify(True)


if __name__ == '__main__':
    app.run(host='localhost', port=3002, debug=True)
