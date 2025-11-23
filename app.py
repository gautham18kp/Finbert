import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments
import matplotlib.pyplot as plt
import pandas as pd
import os
from flask import Flask, request, render_template, send_file
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the prebuilt DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_scores = predictions[0].tolist()
    sentiment_labels = ['negative', 'positive']
    sentiment = sentiment_labels[sentiment_scores.index(max(sentiment_scores))]
    
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_news', methods=['POST'])
def get_news():
    ticker = request.form['ticker']
    base_url = "https://finviz.com/quote.ashx?t="
    url = base_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        
        # Process the news table
        parsed_news = []
        for row in news_table.findAll('tr'):
            title = row.a.get_text()
            date_data = row.td.get_text().split()
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
            parsed_news.append([ticker, date, time, title])
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_news, columns=['ticker', 'date', 'time', 'title'])
        
        # Sentiment Analysis using the prebuilt model
        df['sentiment'] = df['title'].apply(lambda title: predict_sentiment(title))
        
        # Prepare sentiment data for Chart.js
        sentiment_data = df['sentiment'].value_counts().to_dict()
        
        return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values, sentiment_data=sentiment_data)
    except Exception as e:
        return str(e)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(os.path.join('static', filename))

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/contact-us')
def contact_us():
    return render_template('contact-us.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    # You can process the contact form data here (e.g., save to a database or send an email)
    return f"Thank you, {name}. Your message has been received!"

if __name__ == '__main__':
    app.run(debug=True)