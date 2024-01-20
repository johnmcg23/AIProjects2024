import os
import time
from PIL import Image
import cv2
import face_recognition
import requests
import base64
import io
import numpy as np


def fetch_repository_contents(github_token, owner, repo):
    # Set the headers for the API request
    headers = {
        'Authorization': 'token ' + github_token,
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        # Make the API request to get the contents of the repository
        response = requests.get(f'https://api.github.com/repos/{owner}/{repo}/contents', headers=headers)

        # Raise an exception for unsuccessful responses
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository contents: {e}")
        return []


def search_face_in_github(github_token, owner, repo, contents):
    # Iterate through each file in the repository
    for file_info in contents:
        if 'download_url' in file_info and file_info['type'] == 'file':
            # Download the image file
            try:
                image_content = requests.get(file_info['download_url']).content
                img = Image.open(io.BytesIO(image_content))
                img_np = np.array(img)

                # Check if a face is detected in the current image
                current_face_encodings = face_recognition.face_encodings(img_np)
                if len(current_face_encodings) > 0:
                    current_face_encoding = current_face_encodings[0]

                    # Compare the face with all faces in the repository
                    match, filename = compare_face_with_repo(github_token, owner, repo, contents, current_face_encoding)

                    if match:
                        print(f"Match in GitHub repo for this face. Filename: {filename}")
                        return True

            except Exception as e:
                print(f"Error processing image: {e}")

    # No match found in any image in the repository
    print("No match in this repo for this face")
    return False


def compare_face_with_repo(github_token, owner, repo, contents, current_face_encoding):
    # Iterate through each file in the repository
    for file_info in contents:
        if 'download_url' in file_info and file_info['type'] == 'file':
            # Download the image file
            try:
                image_content = requests.get(file_info['download_url']).content
                img = Image.open(io.BytesIO(image_content))
                img_np = np.array(img)

                # Get the face encodings for the image in the repository
                repo_face_encodings = face_recognition.face_encodings(img_np)

                for repo_face_encoding in repo_face_encodings:
                    # Check if the face encodings are different before comparison
                    if not np.array_equal(current_face_encoding, repo_face_encoding):
                        # Set the distance threshold for face matching
                        distance_threshold = 0.6

                        # Compare the faces
                        match = face_recognition.compare_faces(
                            [repo_face_encoding],
                            current_face_encoding,
                            tolerance=distance_threshold
                        )

                        if any(match):
                            # Return the match and filename
                            print(file_info['name'])
                            return True

            except Exception as e:
                print(f"Error processing image: {e}")

    # No match found in any image in the repository
    return False, None


def main():
    github_token, owner, repo = get_github_details()
    contents = fetch_repository_contents(github_token, owner, repo)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Countdown parameters
    countdown_start = 3
    countdown = countdown_start

    # Record the start time
    start_time = time.time()

    while True:
        time.sleep(1)  # Delay execution for 1 second
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
                # Check the face in the GitHub repository
                success = search_face_in_github(github_token, owner, repo, contents)

                cap.release()

                return success  # Return boolean value
        else:
            print('No face detected')
            return False


def get_github_details():
    # Load the github details
    github_token = os.getenv('GITHUB_TOKEN')
    owner = 'johnmcg23'
    repo = 'ImagesForAI'
    return github_token, owner, repo


if __name__ == "__main__":
    main()
