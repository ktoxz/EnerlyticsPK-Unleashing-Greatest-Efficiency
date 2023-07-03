
import json
import numbers
from flask import Blueprint, render_template, request, send_from_directory

generate = Blueprint('generate', __name__)

@generate.route('/generator', methods=['POST', 'GET'])
def generator():
    data = request.form
    dataPass = []
    
    if request.method == 'POST':
        print('GET DATA FROM GENERATE ', data)
        for item in data:
            dataPass.append(float(data.get(item)))
        data_json = json.dumps(dataPass)
        print('DATA PASS ', dataPass)
        return render_template("result.html", args=data_json)

    if request.method == 'GET':
        print('GET DATA:', data)
        
    return render_template("generator.html")

@generate.route('/result', methods=['GET'])
def result():
    return render_template("result.html", boolean=True)

@generate.route('/tfjs_model/<path:filename>')
def serve_tfjs_model(filename):
    return send_from_directory('static/tfjs_model2',filename)