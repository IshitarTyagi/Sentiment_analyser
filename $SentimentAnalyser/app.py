from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html page for user input

# Route for analyzing sentiment
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get text input from the user
    text = request.form['text']
    
    # Perform sentiment analysis
    sentiment = sia.polarity_scores(text)
    
    # Extract sentiment scores
    positive = sentiment['pos']
    neutral = sentiment['neu']
    negative = sentiment['neg']
    
    # Build a response based on sentiment scores
    if positive > 0.5:
        result = "Your sentiment is highly positive! 😊 Keep it up!"
    elif negative > 0.5:
        result = "Your sentiment is quite negative 😞. Hope things get better soon!"
    elif neutral > 0.5:
        result = "Your sentiment is neutral. Stay balanced!"
    else:
        result = "Your sentiment is a bit mixed. Let's stay positive!"

    # Return the result as a response
    return render_template('result.html', positive=positive, neutral=neutral, negative=negative, result=result)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

