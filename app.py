from scipy.spatial import distance

import csv
# from Classification import Results
import cv2
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import normalize
import pandas as pd
from cancellability.feature_rdmtransform import transformMeximumCurvatureRDM as MCRG
# import transformUsingMeximumCurvatureAndRG as MCRG
from classification import KFA
from classification import Projection
import joblib
from flask import Flask, jsonify, request


KeyMat = []
KeyMat1 = []

with open('Key01.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        for i in range(0, len(row)):
            row[i] = float(row[i])
        KeyMat.append(row)

print('Key Matrix Loaded')
KeyMat = np.array(KeyMat)
# KeyMat = KeyMat.T
KeyMat = normalize(KeyMat)

with open('Key02.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        for i in range(0, len(row)):
            row[i] = float(row[i])
        KeyMat1.append(row)

print('Key Matrix 2 Loaded')
KeyMat1 = np.array(KeyMat1)
# KeyMat1 = KeyMat1.T
KeyMat1 = normalize(KeyMat1)


def preprocess(path):
    if path.filename == '':
        print('Empty')
    save_file = "Img_file.jpg"
    path.save(save_file)
    k=100
    N=100
    ids = 0
    model = joblib.load('model.pkl')
    
    Img = cv2.imread("Img_file.jpg", 0)
    Img = np.asarray(Img, dtype=np.float64)
    Img = cv2.resize(Img, (k, N), interpolation=cv2.INTER_CUBIC)
    
    Key = KeyMat[ids]
    Key1 = KeyMat1[ids]
    Key = Key.reshape(k * N, 1)
    Key1 = Key1.reshape(k * N, 1)
    transformedFeatureVector, fvs = MCRG(Img, Key, Key1)  # (100x100,1)

    dataMatrixRG = np.column_stack((transformedFeatureVector)).T
    data = np.array(dataMatrixRG).T
    test_data = ((np.array(data)).T)
    testfeature = Projection.nonlinear_subspace_projection_PhD(test_data, model)
    testfeature = np.nan_to_num(testfeature)
    
    v = np.array(testfeature)
    
    return v


def euclidean_distance(v):
    h = []
    dataset_features = pd.read_csv('train_df.csv')
    dataset_features = dataset_features.loc[:, ~dataset_features.columns.str.startswith('Unnamed: 0')]
    features , label = dataset_features.iloc[:, :-1], dataset_features.iloc[:, -1]
    features = (features.to_numpy()).T
    maxval = 0
    id_l = 0
    maxval2 = [None] * 85
    counterval = -1
    print(np.shape(v))
    for x in range(0, len(label)):
        u = features[:, x]
        print(np.shape(u))
        di = distance.euclidean(u,v)
        #di = scipy.spatial.distance.pdist((u,v), 'euclidean')
        di = np.sqrt((v-u)**2)
        di = di.mean()
 
        
        h.append((di, x))
        
    
    top_distance_image_retrieved = sorted(h)[:1]
    print(top_distance_image_retrieved)
    class_name = int(top_distance_image_retrieved[0][1]/10)+1
    img_id = int(top_distance_image_retrieved[0][1]%10)+1
      
    result = {
        'className': class_name,
        'imgID': img_id
        
    }
    
    print(result)
      

    return result

    

app = Flask(__name__)

@app.route('/')
def home():
    return 'The App is Working'


@app.route('/Img', methods=['POST'])
def emg_classify_handler():
    Img_file = request.files.get('Img')

  
    preprocessing = preprocess(Img_file)
    dist = euclidean_distance(preprocessing)
    
    return jsonify(dist)

if __name__ == '__main__':
    app.run(port = 1000, debug=True, use_reloader=False)
