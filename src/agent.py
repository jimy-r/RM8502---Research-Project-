import ecdsa
import hashlib
import time
import json

class Agent:
    def __init__(self):
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.address = self.public_key.to_string().hex()

    def sign_message(self, message):
        """Signs a message with the agent's private key."""
        message_bytes = json.dumps(message, sort_keys=True).encode('utf-8')
        signature = self.private_key.sign(message_bytes)
        return signature.hex()

def verify_signature(public_key_hex, signature_hex, message):
    """Verifies a signature with a given public key."""
    try:
        public_key = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=ecdsa.SECP256k1)
        message_bytes = json.dumps(message, sort_keys=True).encode('utf-8')
        public_key.verify(bytes.fromhex(signature_hex), message_bytes)
        return True
    except (ecdsa.BadSignatureError, ValueError):
        return False