import json
import re
import nltk
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
# nltk.download('punkt')
# nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from textblob import TextBlob
from tweepy import Client
import tweepy
import credentials


def get_tweets(username):
    overall_sentiment = 0

    try:
        client = Client(bearer_token=credentials.BEARER_TOKEN,consumer_key=credentials.API_KEY,consumer_secret=credentials.API_KEY_SECRET,access_token=credentials.ACCESS_TOKEN,access_token_secret=credentials.ACCESS_TOKEN_SECRET)
        # query = 'donald trump'
        # tweets = client.search_recent_tweets(query=query, max_results=10)
        user = client.get_user(username=username, expansions='pinned_tweet_id', tweet_fields=["created_at","text"])
        fetched_data = client.get_users_tweets(user.data.id, max_results=50)
        for tweet in fetched_data[0]:
            txtarray = ''
            txt = tweet.text
            clean_txt = cleanText(txt) # Cleans the tweet
            txtarray += clean_txt
            stem_txt = TextBlob(stem(clean_txt)) # Stems the tweet
            sent = sentiment(stem_txt) # Gets the sentiment from the tweet
            overall_sentiment += sent
        return [overall_sentiment/len(fetched_data[0]), txtarray]
    except tweepy.TweepyException as e:
        print("Error : " + str(e))
        exit(1)



    # print(tweets)


def cleanText(text):
    text = text.lower()
    # Removes all mentions (@username) from the tweet since it is of no use to us
    text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)

    # Removes any link in the text
    text = re.sub('http://\S+|https://\S+', '', text)

    # Only considers the part of the string with char between a to z or digits and whitespace characters
    # Basically removes punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Removes stop words that have no use in sentiment analysis
    text_tokens = word_tokenize(text)
    text = [word for word in text_tokens if not word in stopwords.words()]

    text = ' '.join(text)
    return text


def stem(text):
    # This function is used to stem the given sentence
    porter = SnowballStemmer('english')
    token_words = word_tokenize(text)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return " ".join(stem_sentence)


def sentiment(cleaned_text):
    # Returns the sentiment based on the polarity of the input TextBlob object
    return cleaned_text.sentiment.polarity



print(get_tweets('pycon'))

def personality():
    status_data = pd.read_csv("mypersonality_final.csv")
    status_data = status_data.dropna()
    status_data = status_data.drop(['BROKERAGE', 'BETWEENNESS', 'NBROKERAGE',
                                    'NBETWEENNESS', 'DENSITY', 'TRANSITIVITY', 'NETWORKSIZE'], axis=1)

    items = ['cEXT', 'cNEU', 'cOPN', 'cAGR', 'cCON']
    for item in items:
        status_data[item] = status_data[item].map({'y': 1.0, 'n': 0.0}).astype(int)

    train, test = train_test_split(status_data, test_size=0.3)
    X_train = train['STATUS']
    X_test = test['STATUS']
    SVC_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
                    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ])
    [sentiment, user_data] = get_tweets('pycon')
    stemm = TextBlob(stem(user_data)) # Stems the tweet
    # print(stemm.shape)
    print(train['cEXT'].head(5))

    for category in items:

        print('... Processing {}'.format(category))
        # train the model using X_dtm & y
        SVC_pipeline.fit(X_train, train[category])
        # compute the testing accuracy
        prediction = SVC_pipeline.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
        print('{}'.format(prediction))



personality()


# class TwitterClient():
#     def  __init__(self, twitter_user=None):
#         self.auth = TwitterAuthenticator().authenticate_app()
#         self.twitter_client = API(self.auth)

#         self.twitter_user = twitter_user

#     def get_user_timeline_tweets(self, num_tweets):
#         tweets = []
#         for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
#             tweets.append(tweet)
#         return tweets

# class TwitterAuthenticator():
#     def authenticate_app(self):
#         auth = OAuthHandler(credentials.API_KEY, credentials.API_KEY_SECRET)
#         auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)
#         return auth

# class StdOutListener(Client):
#     def on_data(self, data):
#         print(data)
#         return True

#     def on_error(self, status):
#         if status == 420:
#             return False
#         print(status)


# if __name__=="__main__":
#     client = Client(bearer_token=credentials.BEARER_TOKEN,consumer_key=credentials.API_KEY,consumer_secret=credentials.API_KEY_SECRET,access_token=credentials.ACCESS_TOKEN,access_token_secret=credentials.ACCESS_TOKEN_SECRET)
#     # query = 'donald trump'
#     # tweets = client.search_recent_tweets(query=query, max_results=10)
#     user = client.get_user(username='pycon', expansions='pinned_tweet_id', tweet_fields=["created_at","text"])
#     fetched_data = client.get_users_tweets(user.data.id, max_results=50)
#     for item in fetched_data[0]:

#         print(item.text)
#         print('\n')

    # twitter_client = TwitterClient('pycon')
    # print(twitter_client.get_user_timeline_tweets(2))
    # stream = StdOutListener(bearer_token=credentials.Bearer_Token, return_type=['json'])


