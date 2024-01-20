from flask import Flask, jsonify, request
import cv2
import os
from time import time, strftime
import requests
import base64


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam
    cap = cv2.VideoCapture(0)

    # Countdown parameters
    countdown_start = 3
    countdown = countdown_start

    # Record the start time
    start_time = time()

    while True:
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
            if time() - start_time >= 1:
                countdown -= 1
                start_time = time()

            # If the countdown has finished
            if countdown < 0:
                # Save the image with the exact time stamp as the name
                current_time = strftime("%Y-%m-%d_%H-%M-%S")  # Format the time as a string
                filename = current_time + '.png'
                cv2.imwrite(filename, img)

                # Call the new function to upload the image to GitHub
                success = upload_to_github(filename)

                cap.release()

                return success  # Return boolean value


def upload_to_github(filename):
    # Convert the image to base64
    with open(filename, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode()

    # Set your personal access token, repo, and owner
    token = os.getenv('GITHUB_TOKEN')
    owner = 'johnmcg23'
    repo = 'ImagesForAI'

    # Set the headers for the API request
    headers = {
        'Authorization': 'token ' + token,
        'Accept': 'application/vnd.github.v3+json'
    }

    # Set the data for the API request
    data = {
        'message': 'Add image',
        'content': img_base64
    }

    # Make the API request
    response = requests.put(f'https://api.github.com/repos/{owner}/{repo}/contents/{filename}', headers=headers,
                            json=data)

    # Delete the file
    if os.path.exists(filename):
        os.remove(filename)

    # Check the response
    if response.status_code == 201:
        print("Face stored")
        return True
    else:
        print("Face NOT stored")
        return False


if __name__ == "__main__":
    main()
