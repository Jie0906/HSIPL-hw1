from flask import Flask, url_for, request, render_template, redirect, flash
from flask_sqlachemy import SQLAlchemy
import os

app = Flask(__name__,
    static_folder = "public",
    static_url_path = "/public/"
        )

@app.route("/")
def home_page():
    return render_template('index.html')

app.run(port = 3000)