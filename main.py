from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import os
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image as PILImage
import psycopg2
import uuid
from IPython.display import  display


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'stored-faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

conn = psycopg2.connect("<URI LINK>") #Insert your Aiven DataBase URI in here
cur = conn.cursor()


cur.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'pictures'
    );
""")
table_exists = cur.fetchone()[0]

# If the table does not exist, create it
if not table_exists:
    # Create the 'pictures' table
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE pictures (
            picture TEXT PRIMARY KEY,
            embedding VECTOR(768)
        );
    """)
    conn.commit()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Opening the image and calculating embeddings
            img = PILImage.open(file_path)
            ibed = imgbeddings()
            embedding = ibed.to_embeddings(img)

            # Inserting the filename and embedding into the database
            cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)",
                        (file.filename, embedding[0].tolist()))
            conn.commit()

            # Process the image to extract faces
            extracted_face = process_image(file_path)

            return render_template('upload.html', uploaded=True, filename=file.filename, face_image=extracted_face)
    return render_template('upload.html', uploaded=False)


@app.route('/find', methods=['GET', 'POST'])
def find_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Convert uploaded image to embeddings
            img = PILImage.open(file_path)
            ibed = imgbeddings()
            embedding = ibed.to_embeddings(img)

            # Search for the image in the database
            cur = conn.cursor()
            string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
            cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
            rows = cur.fetchall()
            cur.close()

            result_image = None
            if rows:
                result_image = rows[0][0]  # Assuming the first column is the filename

            return render_template('result.html', uploaded_image=file.filename, result_image=result_image)
    return render_template('find.html')

def process_image(file_path):
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)
    if haar_cascade.empty():
        raise IOError("Haar Cascade xml file not found or failed to load.")
    img = cv2.imread(file_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
    )
    for x, y, w, h in faces:
        cropped_image = gray_img[y: y + h, x: x + w]
        target_file_name = os.path.join(UPLOAD_FOLDER, 'extracted_face.jpg')
        cv2.imwrite(target_file_name, cropped_image)
        return target_file_name
    return None



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/stored-faces/<filename>')
def stored_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
