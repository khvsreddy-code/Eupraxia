"""Simple test to call the multimodal server's image endpoint and download the result.
Usage: run this after you start the multimodal server (uvicorn app:app --port 9000).
"""
import requests
import os

SERVER = "http://127.0.0.1:9000"

def main():
    payload = {
        "prompt": "A cinematic sci-fi nebula, ultra-detailed, concept art",
        "width": 512,
        "height": 512,
        "num_inference_steps": 12,
        "guidance_scale": 7.5,
        "use_remote": True,
        "provider": "auto"
    }
    try:
        resp = requests.post(f"{SERVER}/generate/image", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        print('Generation response:', data)
        url = data.get('url')
        if url:
            # download
            full = SERVER + url
            r2 = requests.get(full, stream=True)
            if r2.status_code == 200:
                out = os.path.join(os.getcwd(), 'multimodal_test_output.png')
                with open(out, 'wb') as f:
                    for chunk in r2.iter_content(1024):
                        f.write(chunk)
                print('Saved to', out)
    except Exception as e:
        print('Test failed:', e)

if __name__ == '__main__':
    main()
