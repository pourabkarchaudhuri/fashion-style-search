import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import codecs, json 
import base64
import time
from operator import itemgetter
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_URL = "http://52.172.32.166:4000/"
# BASE_URL = "http://localhost:4000/"

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
img_name = []

for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    # print('static/img/' + os.path.splitext(os.path.basename(feature_path))[1] + '.jpg')
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
    img_name.append(os.path.splitext(os.path.basename(feature_path))[0])


@app.route('/', methods=['GET', 'POST'])
def index():
    scores = {}
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static\\uploaded\\" +  str(int(round(time.time() * 1000))) + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:10] # Top 8 results
        print(ids)
        for id in ids:
            if (0 <= dists[id] <= 2):
                scores = {(dists[id], img_paths[id]) for id in ids}
                names = [[dists[id], img_name[id]] for id in ids]

        # print("Scores : ", scores)
        # print("Names : ", names)
        filteredScores = list()
        filteredNames = list()
        if len(scores) == 0:
            return render_template('index.html', query_path=uploaded_img_path, scores="None", names=[])
        for idx, x in enumerate(scores):
            # x(len(x.split('/')))
            t = x[1].split('/')
            filename = t[len(t) - 1].split('.')[0]
            for index, y in enumerate(names):
                if (y[1] == filename):
                    y[1] = ''.join([i for i in y[1] if not i.isdigit()])
                    filteredScores.append(x)
                    filteredNames.append(y)
                    break
        filteredNames = sorted(filteredNames, key=itemgetter(0))
        filteredScores = sorted(filteredScores, key=itemgetter(0))

        return render_template('index.html', query_path=uploaded_img_path, scores=filteredScores, names=filteredNames)
        
    else:
        return render_template('index.html')


@app.route('/img/<string:ip>')
def img(ip):
    image = "static/img/"+ip
    return send_file(image, mimetype='image/gif')


@app.route('/recognize', methods=['POST'])
def post_example():
    print(request)
    if not request.headers.get('Content-type') is None:
        if(request.headers.get('Content-type').split(';')[0] == 'multipart/form-data'):
            if 'image' in request.files.keys():
                print("inside get image statement")
                file = request.files['image']
                img = Image.open(file.stream)  # PIL image
                uploaded_img_path = "static\\uploaded\\" +  str(int(round(time.time() * 1000))) + "_" + file.filename
                img.save(uploaded_img_path)
                #print (img)
                query = fe.extract(img)
                dists = np.linalg.norm(features - query, axis=1)  # Do search
                ids = np.argsort(dists)[:10] # Top 8 results
                data ={ "details" : []}
                def add_info(info):
                    data["details"].append(info)
                for id in ids:
                    info = {}
                    info["score"]=str(dists[id])
                    info["path"]=BASE_URL + img_paths[id]
                    info["name"]=img_name[id]
                    add_info(info)
                return jsonify(data)
            else:
                return jsonify(get_status_code("Invalid body", "Please provide valid format for Image 2")), 415

        elif(request.headers.get('Content-type') == 'application/json'):
            if(request.data == b''):
                return jsonify(get_status_code("Invalid body", "Please provide valid format for Image")), 415
            else:
                body = request.get_json()
                if "image_string" in body.keys():
                    str_image = body['image_string']
                    # str_image = img_string.split(',')[1]
                    imgdata = base64.b64decode(str_image)
                    img = "static\\uploaded\\" +  str(int(round(time.time() * 1000))) + "image_file.jpg"
                    with open(img, 'wb') as f:
                        f.write(imgdata)

                    image=Image.open(img)
                    #print (image)
                    #return send_file(img, mimetype='image/gif')

                    query = fe.extract(image)
                    dists = np.linalg.norm(features - query, axis=1)  # Do search
                    ids = np.argsort(dists)[:10] # Top 8 results
                    data ={ "details" : []}
                    def add_info(info):
                        data["details"].append(info)
                    for id in ids:
                        info = {}
                        info["score"]=str(dists[id])
                        info["path"]=BASE_URL + img_paths[id]
                        info["name"]=img_name[id]
                        add_info(info)
                    return jsonify(data)

        else:
            return jsonify(get_status_code("Invalid header", "Please provide correct header with correct data")), 415

    else:
        return jsonify(get_status_code("Invalid Header", "Please provide valid header")), 401

def get_status_code(argument, message):
    res = {
        "error": {
            "code": argument,
            "message": message
        }
    }
    return res

if __name__=="__main__":
    app.run(host="0.0.0.0", port=4000)
