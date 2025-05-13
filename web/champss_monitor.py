from flask import Flask, url_for
from flask import Response
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import time
import requests
import traceback
import threading
from . import dir_loader
from . import login as login_hdl
from . import api as api_hdl
from ..backend.utils.utils import utils

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

    x_forwarded_proto = request.headers.get('X-Forwarded-Proto')
    if  x_forwarded_proto == 'https':
        request.url = request.url.replace('http://', 'https://')
        request.url_root = request.url_root.replace('http://', 'https://')
        request.host_url = request.host_url.replace('http://', 'https://')
        request.base_url = request.base_url.replace('http://', 'https://')

@app.after_request
def after_request(response):
    global app
    app.last_request = time.time()
    return response

@app.route('/', methods=['GET'])
def index():
    tag_filter = "all"
    if 'tag' in request.args:
        tag_filter = request.args['tag']

    date_filter = "none"
    if 'dates' in request.args:
        date_filter = request.args["dates"]

    warning_filter = 0
    if 'warning' in request.args:
        warning_filter = int(request.args["warning"])

    return render_template(
        'index.html',
        app=app,
        sources=app.sources,
        request=request, 
        tag_filter=tag_filter, 
        date_filter=date_filter, 
        warning_filter=warning_filter, 
        show_sidebar=True
    )

@app.route('/plots')
def plots():
    return render_template(
        'plots.html',
        app=app,
        sources=app.sources,
        request=request, 
        show_sidebar=True
    )

@app.route('/ephemeris')
def ephemeris():
    tag_filter = "all"
    if 'tag' in request.args:
        tag_filter = request.args['tag']

    return render_template(
        'ephemeris.html',
        app=app,
        sources=app.sources,
        request=request, 
        f02p0=utils.utils.f02p0,
        f12p1=utils.utils.f12p1,
        round=round, 
        deg2dms=utils.utils.deg2dms, 
        min=min,
        max=max, 
        tag_filter=tag_filter, 
        show_sidebar=True
    )

@app.route('/assets/<path:path>')
def send_assets(path):
    return app.send_static_file(path)

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
        show_msg=show_msg, 
        show_sidebar=True
    )

@app.route('/diagnostic/<source_id>')
def diagnostic(source_id):
    show_sidebar = True
    if 'preview' in request.args:
        show_sidebar = False

    return render_template(
        'diagnostic.html',
        app=app,
        source_id=source_id,
        sources=app.sources,
        request=request, 
        show_sidebar=show_sidebar, 
        round=round, 
        deg2dms=utils.utils.deg2dms,
    )

@app.route('/diagnostic/<source_id>/pulse_profiles')
def pulse_profiles(source_id):
    return render_template(
        'pulse_profiles.html',
        app=app,
        source_id=source_id,
        sources=app.sources,
        request=request, 
        show_sidebar=False, 
        obs_mjds = app.sources[source_id].get_profile_mjds()
    )

@app.route('/diagnostic/<source_id>/pulse_profiles/<filename>')
def pulse_profiles_get_data(source_id, filename):
    return app.sources[source_id].get_profile_data(filename)

@app.route('/diagnostic/<source_id>/pdf')
def diagnostic_pdf(source_id):
    res = Response(open(app.sources[source_id].pdf, 'rb').read())
    res.headers['Content-Type'] = 'application/pdf'
    return res

@app.route('/diagnostic/<source_id>/pdf/')
def diagnostic_pdf_(source_id):
    return diagnostic_pdf(source_id)

@app.route('/diagnostic/<source_id>/parfile')
def parfile(source_id):
    res = Response(app.sources[source_id].get_parfile())
    res.headers['Content-Type'] = 'text/plain'
    res.headers['Content-Disposition'] = f'attachment; filename="champss_timing_{source_id}.par"'
    return res

@app.route('/diagnostic/<source_id>/parfile/')
def parfile_(source_id):
    return parfile(source_id)

@app.route('/diagnostic/<source_id>/timfile')
def timfile(source_id):
    res = Response(app.sources[source_id].get_timfile())
    res.headers['Content-Type'] = 'text/plain'
    res.headers['Content-Disposition'] = f'attachment; filename="champss_timing_{source_id}.tim"'
    return res

@app.route('/diagnostic/<source_id>/timfile/')
def timfile_(source_id):
    return timfile(source_id)

@app.route('/diagnostic/<source_id>/dbfile')
def sqlite3(source_id):
    res = Response(open(app.sources[source_id].source_dir + "/champss_timing.sqlite3.db", 'rb').read())
    res.headers['Content-Type'] = 'application/octet-stream'
    res.headers['Content-Disposition'] = f'attachment; filename="champss_timing_{source_id}.sqlite3.db"'
    return res

@app.route('/diagnostic/<source_id>/dbfile/')
def sqlite3_(source_id):
    return sqlite3(source_id)

@app.route('/diagnostic/<source_id>/dealias/diagnostics')
def dealias_diagnostics(source_id):
    res = Response(open(app.sources[source_id].source_dir + "/dealias/diagnostic.pdf", 'rb').read())
    res.headers['Content-Type'] = 'application/pdf'
    res.headers['Content-Disposition'] = f'attachment; filename="champss_timing_{source_id}_dealias_diagnostic.pdf"'
    return res

@app.route('/diagnostic/<source_id>/dealias/diagnostics/')
def dealias_diagnostics_(source_id):
    return dealias_diagnostics(source_id)

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