# Sentiment-Analysis-Model & API
This is the source code thai/eng sentiment analysis model & API use for Education only

## Project structure
```text
    . root
    ├── datasets                      # Datasets TH & EN
    ├── datasets[DEV]                 # Datasets TH & EN for develop
    ├── deploy                        # Deploy model folder
    |    └── main.py                       # Fast API Route
    ├── models                        # Model Sentiment analysis folder
    |   ├── modelEN.ipynb                  # model sentiment analysis thai language
    |   └── modelTH.ipynb                  # model sentiment analysis english language
    ├── savedmodel                    # Check point save model & tokenizer for loadmodel
    └── webscrapping                  # Web scrapping folder & export comments.csv
```
