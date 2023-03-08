from langdetect import detect

print(detect("My name is peeter"))

from pythainlp.tokenize import word_tokenize
text = "เคยอม ตา กลม"
list_word = word_tokenize(text)
print(list_word)
