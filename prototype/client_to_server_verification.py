# Function to verify the password with the server
import requests


def verify_password(password,url_verify_password):
    data = {"password": password}
    try:
        response = requests.post(url_verify_password, json=data)
        if response.status_code == 200:
            return True, response.json().get("message", "Password verified successfully")
        else:
            return False, response.json().get("error", "Password verification failed")
    except requests.exceptions.RequestException as e:
        return False, f"Error communicating with server: {e}"

def verify_server(server_ip):
    """
    Sends a request to check if the server is reachable.
    """
    url = f"http://{server_ip}:5000/verify-address"  # Default server endpoint for verification
    payload = {"server_ip": server_ip}

    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return True, response.json()["message"]
        else:
            return False, response.json()["message"]
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"
