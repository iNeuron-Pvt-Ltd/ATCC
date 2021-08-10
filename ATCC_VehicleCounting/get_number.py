import requests
import json
import os
from base64 import b64encode
import string
import pymongo
import datetime
client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

api_key = 'AIzaSyABlj0DEz8_-XbW-hA2vN4GTJpnT6gdzsw'
ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
plate = []
code = ['AP', 'AN', 'AR', 'AS', 'BR', 'CH', 'CR', 'DL', 'DD', 'DN', 'GA', 'GJ', 'HR', 'HP', 'JK', 'KL',
        'KA','LD','MH', 'MN', 'ML', 'MZ', 'MP', 'NL', 'PB', 'RJ', 'SK', 'TN', 'TR', 'TG', 'JH', 'PY',
        'UA', 'UK', 'UP','OR','OD', 'WB']
alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#tme = []

def database(t,N):
    mydb = client["VehicleRecords"]
    inf = mydb.numberplate
    record = {'time':t,'number':N}
    inf.insert_one(record)


def makeImageData(imgpath):
    img_req = None
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def requestOCR(url, api_key, imgpath):
    imgdata = makeImageData(imgpath)
    response = requests.post(ENDPOINT_URL, data = imgdata, params = {'key': api_key},
                             headers = {'Content-Type': 'application/json'})
    return response

def Time():
    time = datetime.datetime.now()
    tm = str(time.hour ) +': ' +str(time.minute ) +': ' +str(time.second)
    return tm


def remove(path):
    os.remove(path)


def Noplate_extract(image):
    try:
        result = requestOCR(ENDPOINT_URL, api_key, image)
        txt = result.json()['responses'][0]['textAnnotations'][0]['description']
        words = txt.split()
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        txt = ('').join(stripped[:10])
        t = txt.split()
        T = t[0][0] + t[0][1]
        for i in range(len(alpha)):
            if (((t[0][2] and t[0][3])) not in alpha[i]) and ((t[0][4]) in alpha[i]) and (t[0][-4:] not in alpha[i]):
                for i in range(len(code)):
                    if T in (code[i]) and (len(stripped[0]) >= 7) and (len(stripped[0]) <= 10) and txt not in plate:
                        plate.append(txt)
                        print(plate)
            #tme.append(Time())
                        database(Time(), txt)


        #pd.DataFrame({'time' :tme ,'number' :plate}).to_csv('data/number.csv')
            #Df.to_csv('data/number.csv')
        #return Df
    except:
        pass
