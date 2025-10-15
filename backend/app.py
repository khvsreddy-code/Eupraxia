from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load a pre-trained model (quantized)
model_name = "meta-llama/Llama-3.3-8B-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)

# Load your data for RAG (simplified)
with open("evolution_data/smoke_10.jsonl", "r") as f:
    lines = f.readlines()
data = [eval(line) for line in lines]  # Assumes JSONL format

def generate_response(prompt):
    # Simple RAG: Match prompt to data
    context = " ".join([x["text"] for x in data if prompt.lower() in x["text"].lower()][:2])
    full_prompt = f"{context}\n{prompt}" if context else prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    response = generate_response(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)