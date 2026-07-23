import os
import requests

FIREBASE_DB_URL = ""

for path in ["collisions", "violations", "status", "results"]:
    url = f"{FIREBASE_DB_URL}/{path}.json"
    r = requests.delete(url, timeout=10)
    print(path, r.status_code, r.text)
