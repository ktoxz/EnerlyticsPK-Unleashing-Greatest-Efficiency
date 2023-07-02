
import numbers
from flask import Blueprint, render_template, request

generate = Blueprint('generate', __name__)

@generate.route('/generator', methods=['POST', 'GET'])
def generator():
    data = request.form
    print('GET DATA FROM GENERATE ', data)
    if request.method == 'POST':
        activePower = request.form.get('activepower')
        reactivePower = request.form.get('reactivepower')
        voltage = request.form.get('voltage')
        intensity = request.form.get('intensity')
        return render_template("result.html", args=[activePower, reactivePower, voltage, intensity])

        
    return render_template("generator.html")

@generate.route('/result', methods=['GET'])
def result():

    return render_template("result.html", boolean=True)