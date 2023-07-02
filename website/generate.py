
import numbers
from flask import Blueprint, render_template, request

generate = Blueprint('generate', __name__)

@generate.route('/generator', methods=['POST', 'GET'])
def generator():
    data = request.form
    dataPass = []
    print('GET DATA FROM GENERATE ', data)
    if request.method == 'POST':
        for item in data:
            dataPass.append(float(data.get(item)))
        print('DATA PASS ', dataPass)
        return render_template("result.html", args=dataPass)

        
    return render_template("generator.html")

@generate.route('/result', methods=['GET'])
def result():

    return render_template("result.html", boolean=True)