import os
from flask import Flask, request, render_template, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model
from moleimages import MoleImages
import tensorflow as tf
import random


UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 2048 * 2048

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #os.system('rm tmp/test.png')
            filename = 'test.' + filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict', filename = filename))
            # return redirect(url_for('uploaded_file',
            #                         filename=filename))
    return render_template('upload.html')  #upload

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/predict/<filename>')
def predict(filename):
    #with open('data/model.pkl', 'rb') as f:
    #    model = pickle.load(f)
    #text = [str(request.form['user_input'])]
    #result = model.predict(text)
    mimg = MoleImages()
    path_to_file = app.config['UPLOAD_FOLDER'] + '/' + filename
    X = mimg.load_image(path_to_file)
    global graph
    with graph.as_default():
        y_pred = model.predict(X)[0,0]
        print(y_pred,type(y_pred))
    if y_pred > 0.9:
        result = 'High Risk'
        print(result)
    elif (y_pred <= 0.9 and y_pred > 0.5):
        result = 'Medium Risk'
        print(result)
    else:
        result = 'Low Risk'

    page = '{0}'
    print (path_to_file)
    print (y_pred)
    print(y_pred.shape)

    #result = 'Low Risk'
    path_to_file = '/uploads/' + filename + '?' + str(random.randint(1000000,9999999))
    return render_template('index.html', image = path_to_file, scroll = 'features', data = page.format(result))



if __name__ == '__main__':
    global model
    model = load_model('models/BM_VA_VGG_FULL_DA.hdf5')
    graph = tf.get_default_graph()
    app.run(host='0.0.0.0', port=7000, debug=True)
