import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import nltk

# Download nltk resources
nltk.download('stopwords', quiet=True)

app = Flask(__name__, template_folder='../templates', static_folder='../static')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        raw_text = request.form['review']

        # Instantiate PorterStemmer
        p_stemmer = PorterStemmer()

        # Remove HTML
        review_text = BeautifulSoup(raw_text, 'html.parser').get_text()

        # Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        # Convert words to lower case and split each word up
        words = letters_only.lower().split()

        # Convert stopwords to a set
        stops = set(stopwords.words('english'))

        # Adding on stopwords
        stops.update(['app','shopee','shoppee','item','items','seller','sellers','bad'])

        # Remove stopwords
        meaningful_words = [w for w in words if w not in stops]

        # Stem words
        meaningful_words = [p_stemmer.stem(w) for w in meaningful_words]

        # Join words back into one string, with a space in between each word
        final_text = pd.Series(" ".join(meaningful_words))

        # Generate predictions
        pred = model.predict(final_text)[0]

        if pred == 1:
            output = "Negative"
        else:
            output = "Positive"

        return render_template('index.html', prediction_text='{} sentiment predicted'.format(output))
        
    return render_template('index.html')

# For local development
if __name__ == "__main__":
    app.run(debug=True)

# This is necessary for Vercel
app = app