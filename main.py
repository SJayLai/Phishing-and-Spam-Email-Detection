from flask import Flask, render_template, request
from model import network
from feature_extractor import feature_extract_from_url
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def load_model():
    print('[model loading]')
    model = network(45)
    filename = 'ckpt_70.pth'
    model.load_state_dict(torch.load(filename))
    return model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',display="none")

@app.route("/submit", methods=['POST'])
def submit():

    def preprocessing(x):
        s = StandardScaler()
        x_scale = s.fit_transform(x)
        return x_scale

    phishing_txt = ["yes","no"]
    url = request.values['url'] 
    features = np.array(feature_extract_from_url(url)).astype(np.float32).reshape(1,-1)

    print(features.dtype)
    print(preprocessing(features))

    features = torch.tensor(preprocessing(features))
    
    print(features.dtype)
    
    model = load_model()
    model.eval()
    with torch.no_grad():
        print(model(features))
        _, output = model(features).max(dim=-1)
    phishing = phishing_txt[output]
    return render_template('index.html', url=url, phishing=phishing, display="block")

@app.route("/test")
def test():
    return render_template('test.html')
if __name__ == '__main__':
    app.debug = True
    app.run()