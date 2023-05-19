import uvicorn
from langdetect import detect

# model
from modelTH import predictTH as model_predictTH
from modelTH import predictTextObjectTH as model_predict_ObjectTH
from modelEN import predictEN as model_predictEN
from modelEN import predictTextObjectEN as model_predict_ObjectEN

# Web Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    return {'message': 'Sentiment analysis TH & EN API'}

# predict route
@app.post('/predict')
def predict(payload: dict):
    text = payload.get('text', '')
    try:
        if str(detect(text)) == "th":
            data = model_predictTH(text)
        else:
            data = model_predictEN(text)  
    except:
        data = { "status": "error", "msg": "Invalid payload." }
    return { "status": "success", "data": data }

# predictObject route
@app.post('/predictObject')
def predictObject(payload: dict):
    texts = payload.get('data', '')
    result = {}
    data = texts
    try:
        for comment in data:
            text = comment['Text']
            if str(detect(text)) == "th":
                sentiment, percentage = model_predict_ObjectTH(str(text))
            else:
                sentiment, percentage = model_predict_ObjectEN(str(text))
            comment['Sentiment'] = sentiment
            comment['Percentage'] = percentage
        result = { "status": "success", "data": data }
    except:
        result = { "status": "error", "msg": "Invalid payload." }
    return result

# start server
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)