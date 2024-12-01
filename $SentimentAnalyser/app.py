!Pip install flask nltk
from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__analyser__)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = sia.polarity_scores(text)
    return f"Positive: {sentiment['pos']}, Neutral: {sentiment['neu']}, Negative: {sentiment['neg']}"

if __name__ == '__main__':
    app.run(debug=True)
