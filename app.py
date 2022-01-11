import io
import base64
from flask import Flask, Response, render_template, redirect, url_for, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from electro_modelling.helpers.helpers_audio import image_grid_spectrograms
from electro_modelling.pipelines.inference_pipeline import InferencePipeline


app = Flask(__name__)
pipeline = InferencePipeline()


@app.route('/plot.png')
def plot_png():
    fakes = pipeline.predict(2)
    fig = image_grid_spectrograms(fakes)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/display')
def display_video():
    return redirect(url_for('static', filename='videos/finished_video12.mp4'), code=301)


@app.route('/', methods=['POST'])
def generate():
    try:
        if request.method == 'POST':
            return render_template('project_page.html', nb_sample=request.form["nbSample"])
    except:
        return render_template('project_page.html')


@app.route('/')
def home():
    return render_template('project_page.html')


if __name__ == '__main__':
    app.run()
