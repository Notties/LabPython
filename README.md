# Sentiment-Analysis-Model & API
This is the source code thai/eng sentiment analysis model & API use for Education only

check out this link for source code Sentiment analysis web application -> [Sentiment-Analysis-WebApp ](https://github.com/Notties/Sentiment-Analysis-WebApp)

## Project structure
```text
    . root
    ├── 📂 datasets                   🔸 Datasets TH & EN
    ├── 📂 datasets[DEV]              🔸 Datasets TH & EN for development
    ├── 📂 deploy                     🔸 Deploy model folder
    |   └── 📄 main.py                     🔹 Fast API Route
    ├── 📂 models                     🔸 Model Sentiment analysis folder
    |   ├── 📄 modelEN.ipynb               🔹 model sentiment analysis thai language
    |   └── 📄 modelTH.ipynb               🔹 model sentiment analysis english language
    ├── 📂 savedmodel                 🔸 Check point save model & tokenizer for loadmodel
    └── 📂 webscrapping               🔸 Web scrapping folder & export comments.csv
```

## Resource
### Datasets Thai
- [Github - thai-sentiment-analysis-dataset](https://github.com/PyThaiNLP/thai-sentiment-analysis-dataset)
- [Github - wisesight-sentiment](https://github.com/PyThaiNLP/wisesight-sentiment/)
- [Kaggle - thai-sentiment-analysis-toolkit](https://www.kaggle.com/datasets/rtatman/thai-sentiment-analysis-toolkit)

### Datasets English
- [Kaggle - Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- [Kaggle - Emotion Dataset for Emotion Recognition Tasks](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
