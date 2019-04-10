from flask import Flask
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
from flask import render_template

app = Flask(__name__)

"""
To run:
export FLASK_APP=server.py
export FLASK_DEBUG=1
"""

@app.route('/')
def home_page():
    return render_template('main.html')
    
@app.route('/<num>')
def image(num):
    chart = f'static/images/fig_person_{num}.png'
    face = f'static/images/{num}.png'
    return render_template('image.html',
                          chart = chart,
                          face = face)
