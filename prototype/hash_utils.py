import hashlib
import os

# Password hashing and verification logic
def hash_data(data, salt, iterations=42): #42 hash iterations with salt
    hash_result = (salt + data).encode()
    for _ in range(iterations):
        hash_result = hashlib.sha256(hash_result).digest()
    return hash_result.hex()

# Generate salt and hash from the startup password
def generate_password_hash(password):
    salt = os.urandom(16).hex() #random salt
    hashed_password = hash_data(password, salt)
    return salt, hashed_password
