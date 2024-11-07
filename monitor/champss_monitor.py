from flask import Flask
from flask import Response
from flask import render_template
from flask import request

from . import dir_loader

app = Flask(__name__)
app.sources = None

@app.route('/')
def main():
    return render_template(
        'main.html',
        sources=app.sources,
        request=request
    )

@app.route('/login')
def login():
    return render_template(
        'login.html',
        sources=app.sources,
        request=request
    )

@app.route('/diagnostic/<source_id>')
def diagnostic(source_id):
    return render_template(
        'diagnostic.html',
        source_id=source_id,
        sources=app.sources,
        request=request
    )

@app.route('/diagnostic/<source_id>/pdf')
def diagnostic_pdf(source_id):
    res = Response(open(app.sources[source_id].pdf, 'rb').read())
    res.headers['Content-Type'] = 'application/pdf'
    return res

def run(psr_dir, port, debug=False):
    global app

    if debug:
        app.debug = True
        app.config.update(DEBUG=True)

    with dir_loader.dir_loader(psr_dir) as app.sources:
        app.run(port = port)