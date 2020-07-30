from flask import Flask, render_template, url_for
from forms import Uploadform
import os
from PIL import Image
import random
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c776dfde280ba245'

def save_picture(form_picture):
    random_num = random.randint(999, 999999)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = str(random_num) + f_ext
    picture_path = os.path.join(app.root_path, 'static/pics', picture_fn)
    output_size = (150, 150)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

def detect_mask(fetchimg):

    face_cascade = cv2.CascadeClassifier(app.root_path + '/static/data/haarcascade_frontalface_default.xml')
    new_model = tf.keras.models.load_model('trainmodel')
    frame = cv2.imread(app.root_path + '/static/pics/' + fetchimg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_color = img[y:y + h, x:x + w]
        roi = cv2.resize(roi_color, (80, 80))
        xy = image.img_to_array(roi)
        xi = np.expand_dims(xy, axis=0)
        im = np.vstack([xi])
        classes = new_model.predict(im)

        if classes[0][0] == 1.0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 140, 0), 2)
            cv2.putText(frame, "Wear Mask", (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "No Mask", (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(app.root_path + '/static/pics/' + fetchimg, frame)


@app.route('/', methods=['GET', 'POST'])
def home():
    image_file = ''
    form = Uploadform()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)

            image_file = url_for('static', filename= 'pics/' + picture_file)
            detect_mask(picture_file)

    return render_template("index.html", form=form, img = image_file)

if __name__ == '__main__':
    app.run(debug=True)