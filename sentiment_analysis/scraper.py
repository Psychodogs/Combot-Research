from bs4 import BeautifulSoup
import numpy as np
import requests
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
import torch

# Load FinBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("./finbert_model")
model = BertForSequenceClassification.from_pretrained("./finbert_model")

#Variables
ticker = "AAPL"
company_name = "apple"
current_date = datetime(2025, 3, 25)
RELEVANCE_MULTIPLIER = 10
def FALLOFF_FUNC(x):
    return 0.5**x

#Helper functions
def getSentiment(str):
    return str

#Fetch data from Benzinga (apparently an industry standard)
url = f"https://www.benzinga.com/quote/{ticker}/news"
response = requests.get(url)
html = response.text
sentiment_total = total_articles = 0

#Analyse sentiment
def get_sentiment(text):
    '''
    Returns value between -1 and 1
    where 1 is positive, -1 is negative and 0 is neutral
    '''
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    labels = ["Positive", "Neutral", "Negative"]
    sentiment = labels[torch.argmax(probabilities).item()]
    distribution = probabilities.tolist()
    values = np.array([-1,0,1])
    
    return -np.dot(distribution,values)[0]


#Process data to get headlines
soup = BeautifulSoup(html, "lxml")
articles = soup.find("div", class_="news-content")
for article in articles:
    date = article.find("div", class_="date-heading")
    #Checks if is valid and processes headlines
    if date is not None:
        date_str = date.text
        headlines = article.find_all("li")
        for headline in headlines:
            #Processes headline
            headline_url = headline.find("a")["href"]
            title = headline.find("div",class_="content-title").text
            sentiment = get_sentiment(title)
            print(date_str, "\n", title)
            total_articles += 1
            #Check headline relevance to company
            headline_relevance = title.count(company_name.lower()) + 1
            #Check headline date reference
            article_date = datetime.strptime(date_str, "%A, %B %d, %Y")
            date_difference = -(article_date - current_date).days + 1
            date_relevance = FALLOFF_FUNC(date_difference)
            print(sentiment, headline_relevance, date_relevance, "\n")
            sentiment_total += sentiment * headline_relevance * (date_relevance + 1)


print(f"General Sentiment: {sentiment_total/total_articles}")
