from flask import Flask, render_template, request, send_file, session
from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2
import numpy as np
import re
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")