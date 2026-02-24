# TEXT_CLASSIFICATION_SPAM_DETECTION

# AI Internship Challenge: Text Classification

## Project Overview
A modular Text Classification system designed to identify SMS Spam using Machine Learning.

## Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Categories:** Ham (Normal), Spam.

## Tech Stack
- **Language:** Python 3.x
- **ML:** Scikit-Learn (Logistic Regression + TF-IDF)
- **API:** FastAPI + Uvicorn
- **Data:** Pandas

## Environment

**Get the virtual env first**
python -m venv venv

## How to Run
1. **Install Dependencies:**
   `pip install pandas scikit-learn fastapi uvicorn`
   `pip install fastapi`
   `pip install uvicorn`

3. **Train the Model:**
   `python model_trainer.py` (This generates `model.pkl` and `tfidf.pkl`)
    After training the model also run classifier.py (python classifier.py)

4. **Start the API:**
   `python app.py`

5. **Test it:**
   Open your browser to: `http://127.0.0.1:8000/predict?text=Win a free car now!`
   After opening the browser go to http://127.0.0.1:8000/docs it will open a SPAM CLASSIFIER page and test it via get/predict
   
It's all based on the dataset I got from UCI SMS SPAM Collection you can write whatever you want it will return "HAM" that is not a spam message. Othervise if you try "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's" you will see the prediction equals to SPAM
