from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from app import predict,vak_predict

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    sample_input = ['2','3','6','5','0','1','1', '1', '1', '2', '0', '0','1','1','0','7','13','7','2','7','5']
    result = predict(sample_input)
    print("ANSWERRRRR __ ",result[0])
    return result[0]

@app.route('/careerPredict', methods=['POST'])
def example():
    data = request.json  # Access JSON request body
    # Process the data...
    result = predict(data["data"])
    return result[0]

@app.route('/vakModel', methods=['POST'])
def vak():
    data = request.json 
    print(data["data"]) 
    result = vak_predict(data["data"])
    #response = [result[0][0],np.float64(result[0][1])]
    output = []
    for x in result:
        temp=[]
        temp.append(x[0])
        temp.append(np.float64(x[1]))
        output.append(temp)
    return (output)

if __name__ == '__main__':
    app.run(debug=True)
