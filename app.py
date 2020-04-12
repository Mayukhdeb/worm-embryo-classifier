import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# git -- gave my imports 
import numpy as np
from numpy import moveaxis
import torch
import torch.nn.functional as F
import torch.nn as nn                                                
import cv2

# Some utilites
import numpy as np
from util import base64_to_pil


## classes
classes = ["gastrula", "comma", "fold", "l1" ]

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
### loading model 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool = nn.MaxPool2d(5,5)

        self.pool2 = nn.MaxPool2d(3,3)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.conv3 = nn.Conv2d(10, 10, 3)
        
        self.fc1 = nn.Linear(150, 110)
        self.fc2 = nn.Linear(110, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))      ## removed relu
        return x


class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(3,3)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(4, 8, 5)

        self.fc1 = nn.Linear(704,200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 20)

        self.fc4 = nn.Linear(20, 4)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = (self.fc4(x))      ## removed relu
        return x


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(8, 4)
        self.classifier2 = nn.Linear(4, 4)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        x = self.classifier2(F.relu(x))
        return x
    
cellnet_1 = Net()
cellnet_1.zero_grad()
cellnet_1.load_state_dict(torch.load("models/weights_c1.pth"))
cellnet_1.eval()

cellnet_2 = Net_2()
cellnet_2.zero_grad()
cellnet_2.load_state_dict(torch.load("models/weights_c2.pth"))
cellnet_2.eval()

model = MyEnsemble(cellnet_1, cellnet_2)
model.load_state_dict(torch.load("models/weights_combined.pth"))

model.eval()

print('Yee haw ! Model loaded.  Check http://127.0.0.1:5000/')

## loaded model 


def model_predict(img, model):
    img = img.resize((120, 90)) # git -- my own image size 

    # Preprocessing the image
    x = np.array(img)  ## 3D numpy array 

    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)  ## 2D grayscale now 

    input_tensor =  torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()  ## git -- 4D torch tensor 

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   
    preds = torch.softmax(model(input_tensor, input_tensor).flatten(), dim = 0).detach().numpy()
    return preds  ## np array 


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = classes[list(preds).index(max(list(preds)))]
        result = pred_class
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
