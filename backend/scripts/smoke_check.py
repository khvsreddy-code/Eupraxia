"""
Simple smoke check to call the admin endpoints.
"""
import os
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def main():
    print("Calling /admin/health ...")
    try:
        r = requests.get(BASE + "/admin/health", timeout=5)
        print(r.status_code, r.json())
    except Exception as e:
        print("Health call failed:", e)

    print("Calling /admin/smokecheck ...")
    try:
        r = requests.get(BASE + "/admin/smokecheck", timeout=10)
        print(r.status_code, r.json())
    except Exception as e:
        print("Smokecheck failed:", e)


if __name__ == '__main__':
    main()
