import random
import nltk
import numpy as num
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from ChatBotModel import ChatBotModel
from WordsHelper import WordsHelper

nltk.download("punkt")
nltk.download("wordnet")

wordsHelper = WordsHelper()
ourData = wordsHelper.getData()
lm = wordsHelper.getWordNetLemmatizer()
newWords, ourClasses = wordsHelper.getNewWordsAndClasses()
documentX = []
documentY = []
for intent in ourData["ourIntents"]:
    for pattern in intent["patterns"]:
        documentX.append(pattern)
        documentY.append(intent["tag"])

trainingData = []
outEmpty = [0] * len(ourClasses)
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)

x = num.array(list(trainingData[:, 0]))
y = num.array(list(trainingData[:, 1]))
iShape = (len(x[0]),)
oShape = len(y[0])

model = ChatBotModel.build(iShape, oShape)
md = Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=md,
              metrics=["accuracy"])

H = model.fit(x, y, epochs=35, verbose=1)

model.save("ChatBot.model", save_format="h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 35), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 35), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")
