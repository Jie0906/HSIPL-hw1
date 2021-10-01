from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
import os

pj_path = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///' + \
    os.path.join(pj_path,'data.sqlite')
app.config['SECRET_KEY'] = '0000'

bootstap = Bootstrap(app)
db = SQLAlchemy(app)


