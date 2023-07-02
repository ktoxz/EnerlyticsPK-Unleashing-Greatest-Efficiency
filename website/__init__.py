from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ElectroSquad'
    from .about import about
    from .auth import auth
    from .generate import generate
    app.register_blueprint(about, url_prefix='/')
    app.register_blueprint(generate, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    return app