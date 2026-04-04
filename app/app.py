"""
app.py – Flask web UI for the banking chatbot.

Usage:
    python app/app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, render_template, request

from src.inference import generate_response, generate_base_response

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400

    try:
        bot_response = generate_response(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"response": f"Error generating response: {str(e)}"}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Return both base model and fine-tuned model responses for side-by-side comparison."""
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        # Generate base model response first
        base_resp = generate_base_response(user_message)
        # Then fine-tuned (memory is freed between calls inside the function)
        ft_resp = generate_response(user_message)
        return jsonify({"base": base_resp, "finetuned": ft_resp})
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


if __name__ == "__main__":
    print("[app] Starting Flask on http://127.0.0.1:5000")
    print("[app] Models will load on the first request (~30s wait).")
    app.run(host="127.0.0.1", port=5000, debug=False)
