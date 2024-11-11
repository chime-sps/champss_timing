from flask import Flask, url_for
from flask import Response
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import time
from . import dir_loader
from . import login as login_hdl
from . import api as api_hdl
from .. import utils

app = Flask(__name__)
app.sources = None
app.login = None
app.api = None
app.update = None
app.last_request = 0
app.pipeline_version = utils.utils.get_version_hash()
app.secret_key = utils.utils.get_rand_string()

@app.before_request
def before_request():
    if request.endpoint != 'login' and request.endpoint != 'api':
        if not app.login.has_logged_in():
            return redirect(url_for('login'))


@app.after_request
def after_request(response):
    global app
    app.last_request = time.time()
    return response

@app.route('/')
def index():
    return render_template(
        'index.html',
        app=app,
        sources=app.sources,
        request=request
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    show_msg = False
    if request.method == 'POST':
        if 'password' not in request.form:
            show_msg = "No password provided."
        elif app.login.checker(request.form['password']):
            return redirect(url_for('index'))
        else:
            show_msg = "Wrong password."
    return render_template(
        'login.html',
        app=app,
        sources=app.sources,
        request=request,
        show_msg=show_msg
    )

@app.route('/diagnostic/<source_id>')
def diagnostic(source_id):
    return render_template(
        'diagnostic.html',
        app=app,
        source_id=source_id,
        sources=app.sources,
        request=request
    )

@app.route('/diagnostic/<source_id>/pdf')
def diagnostic_pdf(source_id):
    res = Response(open(app.sources[source_id].pdf, 'rb').read())
    res.headers['Content-Type'] = 'application/pdf'
    return res

@app.route('/diagnostic/<source_id>/pdf/')
def diagnostic_pdf_(source_id):
    return diagnostic_pdf(source_id)

@app.route('/public/api/<endpoint>', methods=['GET', 'POST'])
def api(endpoint):
    return app.api.handle(endpoint, request)

def run(psr_dir, port, password=False, debug=False, update_hdl=None):
    global app

    app.login = login_hdl.login(session, password)
    app.api = api_hdl.api(app)
    app.update = update_hdl

    if debug:
        app.debug = True
        app.config.update(DEBUG=True)

    if app.update != None:
        app.update()

    with dir_loader.dir_loader(psr_dir, app) as app.sources:
        app.run(port = port)