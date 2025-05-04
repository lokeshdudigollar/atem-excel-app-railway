from flask import Flask
from shutil import which
import pytesseract

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask + Tesseract App is Running"

@app.route("/debug-tesseract")
def debug_tesseract():
    path = which('tesseract')
    return f"Tesseract found at: {path or 'it is Not found'}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")