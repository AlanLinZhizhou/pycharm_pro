# encoding utf-8
from flask import Flask
from flask import make_response
from flask import render_template
from flask import request as fr
from catDog import inferdog
import json, base64, os
from flask_uploads import UploadSet, configure_uploads, ALL


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
strUploadPath = 'uploads'

files = UploadSet(strUploadPath, ALL)
app.config['UPLOADS_DEFAULT_DEST'] = ''

configure_uploads(app, files)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/uploadsImg', methods=['POST', 'GET'])
def hello():
        #if fr.method == 'POST':
        aa = fr.form['image']
        img_data = base64.b64decode(aa)
        ff = open('./static/photo/01.jpg', 'wb')
        ff.write(img_data)
        ff.flush()
        ff.close()
        path = basedir + '\\static\\photo\\01.jpg'
        result = inferdog.inference(path)
        a=0
        # print type(result)
        # print result
        # print result[0][0][0],result[0][0][1]
        if result[0][0][0] > result[0][0][1]:
            return "this is a cat"
        if result[0][0][0] < result[0][0][1]:
            return "this is a dog"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
