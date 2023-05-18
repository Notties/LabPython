# model
from modelTH import predict as model_predict
from modelTH import predictTextObject as model_predict_Object
# Web Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# create app
app = FastAPI()
# alow cross origin all origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# index route
@app.get('/')
def index():
    return {'message': 'This is a sentiment analysis model API.'}

# predict route
@app.post('/predict')
def predict(payload: dict):
    text = payload.get('text', '')
    if text:
        data = model_predict(text)
        return { "status": "success", "data": data }
    else:
        return { "status": "error", "msg": "Invalid payload." }

@app.post('/predictObject')
def predictObject(payload: dict):
    texts = payload.get('data', '')
    result = {}
    data = texts
    for comment in data:
        text = comment['Text']
        sentiment, percentage = model_predict_Object(str(text))
        comment['Sentiment'] = sentiment
        comment['Percentage'] = percentage
    result = { "status": "success", "data": data }

    return result

# start server
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)