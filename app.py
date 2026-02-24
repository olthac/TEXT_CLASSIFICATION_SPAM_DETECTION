from fastapi import FastAPI
from classifier import SMSClassifier

app = FastAPI(title="Spam Classifier")
model_service = SMSClassifier()

@app.get("/")
def home():
    return {"status": "API is running. Use /predict?text=your_message"}

@app.get("/predict")
def predict(text: str):
    prediction = model_service.predict(text)
    return {
        "input_text": text,
        "prediction": prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)