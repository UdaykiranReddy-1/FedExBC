import os.path

import requests


def download_global():
    file_path = 'global_model.pth'
    # URL of the Flask server's download endpoint
    url = 'http://localhost:5000/download'  # Update with the actual URL if different

    # Make a GET request to download the file
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in write-binary mode and save the content
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded and saved.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def send_fine_tuned():
    # Path to the .pth file you want to upload
    file_path = 'fine_tuned_model.pth'

    # Endpoint URL
    url = 'http://localhost:5000/upload'

    # Open the file in binary mode and send it
    if not os.path.exists(file_path):
        print("Model training not done yet!!")
        return

    with open(file_path, 'rb') as file:
        files = {'model': file}
        response = requests.post(url, files=files)

    # Print the server's response
    print(response.status_code)
    print(response.json())


def check_blockchain():
    url = 'http://localhost:5000/validate'
    response = requests.get(url)
    print(response.json())


while True:
    print("1. Download global model")
    print("2. Upload fine tuned model")
    print("3. Validate block chain")
    i = input("Enter your choice: ")
    if int(i) == 1:
        download_global()
    elif int(i) == 2:
        send_fine_tuned()
    elif int(i) == 3:
        check_blockchain()
    else:
        print("Invalid choice")
