import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import spacy
import locationtagger
import re

nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class UserDataExtractor:

    def extractUserName(self, text):
        nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
        for nltk_result in nltk_results:
            if type(nltk_result) == Tree:
                name = ''
                for nltk_result_leaf in nltk_result.leaves():
                    name += nltk_result_leaf[0] + ' '
                return name

    def extractDestinition(self, text):

        place_entity = locationtagger.find_locations(text=text)

        print("The countries in text : ")
        print(place_entity.countries)

        print("The states in text : ")
        print(place_entity.regions)

        print("The cities in text : ")
        print(place_entity.cities)

        return place_entity.cities[0]

    def extractDateFromText(self, text):
        print(re.findall(r'\d+\S\d+\S\d+', text))

