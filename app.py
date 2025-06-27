from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, os

app = Flask(__name__)
app.secret_key = "change_this_secret"

model = load_model('model/healthy_vs_rotten.h5')
categories = [
    'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
    'Carrot__Healthy', 'Carrot__Rotten', 'Orange__Healthy',
    'Orange__Rotten', 'Potato__Healthy', 'Potato__Rotten'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            path = os.path.join('static/uploads', f.filename)
            f.save(path)
            img = image.load_img(path, target_size=(150,150))
            arr = image.img_to_array(img)/255.
            arr = np.expand_dims(arr, 0)
            prediction = model.predict(arr)
            idx = np.argmax(prediction)
            if idx >= len(categories):
                label = "Unknown"
                conf = 0
            else:
                label = categories[idx]
                conf = float(np.max(prediction)) * 100

            conf = float(np.max(prediction))*100
            result = {"label": label, "confidence": round(conf,2), "image": path}
    return render_template('predict.html', result=result)

@app.route('/tips')
def tips():
    return render_template('tips.html')

@app.route('/about', methods=['GET','POST'])
def about():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        comment = request.form['comment']
        flash("Thanks for your enquiry—someone will contact you soon.", "success")
    return render_template('about.html')

@app.route('/signin', methods=['GET','POST'])
def signin():
    if request.method=='POST':
        user = request.form['username']
        flash(f"Signed in successfully as {user}!", "success")
    return render_template('signin.html')

if __name__=='__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
