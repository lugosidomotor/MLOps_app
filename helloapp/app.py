from flask import Flask, request, jsonify
import torch
from train import SimpleNN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

model = SimpleNN(input_size=10)  # Adjust input size according to dataset
model.load_state_dict(torch.load("model.pth"))
model.eval()

scaler = StandardScaler()

# Load and fit the scaler
train_data = pd.read_csv("train.csv")
scaler.fit(train_data.iloc[:, :-1])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).astype(np.float32)
    features = scaler.transform([features])
    with torch.no_grad():
        prediction = model(torch.tensor(features))
    return jsonify({'prediction': prediction.item()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
