from flask import Flask
from flask import render_template

from . import dir_loader

app = Flask(__name__)
app.debug = True
app.config.update(DEBUG=True)
app.sources = None

@app.route('/')
def main():
    return render_template(
        'main.html',
        sources=app.sources
    )

@app.route('/diagnostic/<source_id>')
def diagnostic(source_id):
    return render_template(
        'diagnostic.html',
        source_id=source_id,
        sources=app.sources
    )

def run(psr_dir, port):
    global app

    # Load sources
    app.sources = dir_loader.dir_loader(psr_dir)

    # Run app
    app.run(port = port)