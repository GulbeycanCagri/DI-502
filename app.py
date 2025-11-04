import os

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.rag_service import plain_chat, query_document, query_online

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if "question" not in request.form or not request.form["question"]:
        return jsonify({"error": "No question provided"}), 400

    question = request.form["question"]
    use_online_research = request.form.get("online_research") == "true"

    file = request.files.get("document")
    file_path = None

    # Decide the mode of operation
    try:
        if use_online_research:
            # Mode 1: Online Research RAG
            answer = query_online(question)
        elif file and file.filename != "":
            # Mode 2: Document RAG
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            answer = query_document(question, file_path)
        else:
            # Mode 3: Plain Chat
            answer = plain_chat(question)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
