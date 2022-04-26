import requests

req = requests.post(r'http://127.0.0.1:1000/neural')
print(req.json())