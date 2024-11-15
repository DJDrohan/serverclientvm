import requests

def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        response.raise_for_status()  # Raise an error for bad status codes
        ip_data = response.json()
        return ip_data['ip']
    except requests.RequestException as e:
        print(f"Error fetching public IP: {e}")
        return None

# Get and print the public IP address
public_ip = get_public_ip()
if public_ip:
    print(f"Your public IP address is: {public_ip}")
