1. To start use this you need to install all necessary library with its latest versions 
2. Run command in console line <strong>python -m spacy download en_core_web_sm</strong>
3. For manual use, run ChatBot.py, and after it enter messages that have some flight data for example: "I want to go to New York" or "I need plane to New York from 16/10/2022 to 17/11/2022" 
4. In result, you will receive extracted data from message and label for this message
Project files overview:
<br> ChatBotModel.py - file where Neural Network architecture is built
<br>train.py - file where we train model for chatting
<br>UserDataExtractor.py  - file where we extract data from users messages
<br>WordsHelper.py  - file where we have our example dataset and some functions for working with data
<br>FlightOrderDataManager - example file where we can see how it can work in future
<br> plot.png - plot of training results
<br>ChatBot.model - model file that we get after training

<br><br><br>For future improvements we need get more labeled data for training and improvements in data extractor to process more complex user messages