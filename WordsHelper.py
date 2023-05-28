from nltk.stem import WordNetLemmatizer
import string
import nltk


class WordsHelper:
    # this should be replaced with real data in future use setter
    data = {"ourIntents": [
        {"tag": "age",
         "patterns": ["how old are you?"],
         "responses": ["I am 2 years old and my birthday was yesterday"]
         },
        {"tag": "greeting",
         "patterns": ["Hi", "Hello", "Hey"],
         "responses": ["Hi there", "Hello", "Hi :)"],
         },
        {"tag": "goodbye",
         "patterns": ["bye", "later"],
         "responses": ["Bye", "take care"]
         },
        {"tag": "name_enter",
         "patterns": ["My name is John Boba", "I`m Alex Smith", "My name is"],
         "responses": ["I got your name", "Thank you for entering name"]
         },
        {"tag": "name",
         "patterns": ["what's your name?", "who are you?"],
         "responses": ["I have no name yet," "You can give me one, and I will appreciate it"]
         },
        {"tag": "city_from",
         "patterns": ["I need plane from New York", "I leave Kiyv and need plane", "I go from Zhytomyr"],
         "responses": ["I got city you want to go from" "Thank you, I remembered city, you want flight from"]
         },
        {"tag": "city_to",
         "patterns": ["I need plane to New York", "I want to fly to Zhytomyr", "I go to Zhytomyr"],
         "responses": ["I got city you want to go to", "Thank you, I remembered city, you want fly to"]
         },
        {"tag": "date",
         "patterns": ["I need plane to New York from 16/10/2022 to 17/11/2022"],
         "responses": ["I got date you want to go at"]
         }
    ]}

    def getData(self):
        return self.data

    def setData(self, data):
        self.data = data

    def getWordNetLemmatizer(self):
        return WordNetLemmatizer()

    def getNewWordsAndClasses(self):
        classes = []
        newWords = []
        for intent in self.data["ourIntents"]:
            for pattern in intent["patterns"]:
                word_tokens = nltk.word_tokenize(pattern)
                newWords.extend(word_tokens)

            if intent["tag"] not in classes:
                classes.append(intent["tag"])

        newWords = [self.getWordNetLemmatizer().lemmatize(word.lower()) for word in newWords if
                    word not in string.punctuation]

        return sorted(set(newWords)), sorted(set(classes))
