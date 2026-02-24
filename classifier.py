import pickle

class SMSClassifier:
    def __init__(self, model_path='model.pkl', tfidf_path='tfidf.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)

    def predict(self, text):
        text_processed = self.tfidf.transform([text.lower()])
        prediction = self.model.predict(text_processed)
        return "SPAM" if prediction[0] == 1 else "HAM"