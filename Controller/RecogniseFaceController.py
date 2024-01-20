from flask import Flask, jsonify, request
import threading

from Service.RecogniseFaceService import main

app = Flask(__name__)


@app.route('/recognise/faceid', methods=['POST', 'GET'])
def runFacialRecognitionLogin():
    # Get the username from the request
    username = request.json['username']
    print("Gets in method in python")

    # Start the face recognition in a new thread
    threading.Thread(target=main, args=(username,)).start()

    return jsonify('message: Please wait while we scan your face...', 'result')


if __name__ == '__main__':
    app.run(host='localhost', port=3001, debug=True)
