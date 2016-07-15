from flask import Flask, render_template, request, redirect
import dill
import numpy as np
import pandas as pd
import sklearn as sk

import re
from bs4 import BeautifulSoup
from spacy.en import English


app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else: 
        review_text = review
        if request.form.get['score'] == above:
            score_up = 1
        else:
            score_up = 0

    if not review_text or not request.form.get['score']:
        return render_template('error.html')


    p = figure(title='Data from Quandl WIKI set',
    x_axis_label='date', x_axis_type='datetime')
    for (f, c) in zip(features, Spectral4):
        p.line(df['Date'], df[f], line_color=c, legend=ticker + ": " + f)    

    script, div = components(p)
    return render_template('graph.html', script=script, div=div, ticker=ticker)

if __name__ == '__main__':
    model = dill.load(open('full_model'))
    parser = English()
#    app.run(port=33507)
    app.run(host='0.0.0.0')
