from langdetect import detect
import pandas as pd


# text = "สวัสดีเป็นอะไร Hello"
# lang = detect(text)
# print(lang)

df = pd.read_csv("datasum.csv", encoding='utf-8')
langs = detect(df['text'].to_string())
print(langs)