import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_and_save():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(url, header = None, names=['label', 'message'])

    df['message'] = df['message'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2)

    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("Assets saved to disk")

if __name__ == '__main__':
    train_and_save()