from flask import Blueprint, render_template, request

about = Blueprint('about', __name__, static_folder='static')

@about.route('/')
def home():
    return render_template("index.html")

@about.route('/index')
def index():
    return render_template("index.html")

@about.route('/contact')
def contact():
    return render_template("contact.html")

@about.route('/definition')
def definition():
    return render_template("definition.html")



