import ssl
import certifi

print("Ubicación de certificados:", certifi.where())
ssl.create_default_context(cafile=certifi.where())
print("Certificados configurados correctamente.")

