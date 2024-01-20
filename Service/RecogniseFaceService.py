import time
from PIL import Image
import cv2
import face_recognition
import os
import requests
import base64
import io
import numpy as np


def check_in_github(filename):
    # Set your personal access token, repo, and owner
    token = os.getenv('GITHUB_TOKEN')
    owner = 'johnmcg23'
    repo = 'ImagesForAI'

    # Set the headers for the API request
    headers = {
        'Authorization': 'token ' + token,
        'Accept': 'application/vnd.github.v3+json'
    }

    # Make the API request
    response = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents/{filename}', headers=headers)

    # Check the response
    if response.status_code == 200:
        # The file exists in the repo
        # Now we need to compare the face in the current image with the one in the repo
        # First, we load the image from the repo
        img_base64 = response.json()['content']
        img_data = base64.b64decode(img_base64)

        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)

        # Then, we load the current image
        current_img = cv2.imread(filename)
        current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)

        # Check if a face is detected in the current image
        current_face_encodings = face_recognition.face_encodings(current_img_rgb)
        if len(current_face_encodings) == 0:
            print('Authentication face id failed')
            return False

        current_face_encoding = current_face_encodings[0]

        # We get the face encodings for both images
        repo_face_encoding = face_recognition.face_encodings(img_np)[0]

        # And finally, we compare the faces
        match = face_recognition.compare_faces([repo_face_encoding], current_face_encoding)

        if match[0]:
            print('Face matches')
            return True
        else:
            print('Face does not match')
            return False
    else:
        print('Username does not match')
        return False


def main(username):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam
    cap = cv2.VideoCapture(0)

    # Countdown parameters
    countdown_start = 3
    countdown = countdown_start

    # Record the start time
    start_time = time.time()

    while True:
        time.sleep(1)  # Delay execution for 2 seconds
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # If a face is detected
        if len(faces) > 0:
            # Display the countdown
            cv2.putText(img, str(countdown), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # If a second has passed
            if time.time() - start_time >= 1:
                countdown -= 1
                start_time = time.time()

            # If the countdown has finished
            if countdown < 0:
                # Save the image with the username
                filename = username + '.png'
                cv2.imwrite(filename, img)

                # Call the new function to check the image in GitHub
                success = check_in_github(filename)

                # Delete the local file
                if os.path.exists(filename):
                    os.remove(filename)

                cap.release()

                return success  # Return boolean value
        else:
            print('No face detected')
            return False


if __name__ == "__main__":
    main()
