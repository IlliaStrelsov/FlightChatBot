import random
import nltk
import numpy as num
from WordsHelper import WordsHelper
from UserDataExtractor import UserDataExtractor
from keras.models import load_model


def TokenizeText(text):
    newTokens = nltk.word_tokenize(text)
    newTokens = [lm.lemmatize(word) for word in newTokens]
    return newTokens


def wordBag(text, vocab):
    newTokens = TokenizeText(text)
    bagOfWords = [0] * len(vocab)
    for w in newTokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOfWords[idx] = 1
    return num.array(bagOfWords)


def Pclass(text, vocab, labels):
    bagOfWords = wordBag(text, vocab)
    result = model.predict(num.array([bagOfWords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(result) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


def getRes(firstList, fJson):
    tag = firstList[0]
    listOfIntents = fJson["ourIntents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            return result


wordsHelper = WordsHelper()
ourData = wordsHelper.getData()
lm = wordsHelper.getWordNetLemmatizer()
newWords, ourClasses = wordsHelper.getNewWordsAndClasses()
model = load_model("ChatBot.model")
print("To start conversation please enter some message:\n")

while True:
    newMessage = input("")
    intents = Pclass(newMessage, newWords, ourClasses)
    print(intents)
    if intents[0] == 'name_enter':
        print(UserDataExtractor().extractUserName(newMessage))
    elif intents[0] == 'city_from' or intents[0] == 'city_to':
       UserDataExtractor().extractDestinition(newMessage)
    elif intents[0] == 'date':
        print(UserDataExtractor().extractDateFromText(newMessage))
    ourResult = getRes(intents, ourData)
    print(ourResult)
