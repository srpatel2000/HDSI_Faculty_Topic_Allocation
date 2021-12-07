# lda model

import pandas as pd
import sys
import pickle

# for text preprocessing
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import *
import gensim
from gensim.utils import simple_preprocess

# import vectorizers
from sklearn.feature_extraction.text import CountVectorizer

# import numpy for matrix operation
import numpy as np

# import LDA from sklearn
from sklearn.decomposition import LatentDirichletAllocation

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# import plotly.graph_objects as go
# from dash.dependencies import Input, Output, State

# to suppress warnings
from warnings import filterwarnings
filterwarnings('ignore')


def main(argv):
    
    test_data = pd.read_csv('test/testdata/test_data.csv')

    def lemmatize_stemming(text):
        return WordNetLemmatizer().lemmatize(text, pos='v')

    def preprocess_abstract(text):
        result = []
        redundant = ['abstract', 'purpose', 'paper', 'goal', 'usepackage', 'cod']
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
                result.append(lemmatize_stemming(token))
        return " ".join(result)


    def data_cleaning(input_data):

        data = input_data
        data = data[data['year'] > 2014]
        data["abstract"].fillna(data["title"], inplace=True) # if no abstract, replace w/ title of article

        def standardize_abstract(abstract):
            abstract = abstract.replace('\n', ' ')
            abstract = abstract.replace('  ', ' ')
            abstract = abstract.replace('-', ' ')
            abstract = abstract.replace('.', '')
            abstract = abstract.replace(':', '')
            abstract = abstract.replace(';', '')
            abstract = abstract.replace(',', '')
            abstract = abstract.replace('"', '')
            abstract = abstract.lower()
            return abstract

        def standardize_title(title):
            title = title.replace('\n', ' ')
            title = title.replace('  ', ' ')
            title = title.replace('-', ' ')
            title = title.replace('.', '')
            title = title.replace(':', '')
            title = title.replace(';', '')
            title = title.replace(',', '')
            title = title.replace('"', '')
            title = title.lower()
            return title

        data['year'] = data['year'].astype(int)
        data['abstract'] = [standardize_abstract(text) for text in data['abstract']]
        data['title_standardized'] = [standardize_title(text) for text in data['title']]
        data['abstract_processed'] = data['abstract'].apply(preprocess_abstract)
        data.drop_duplicates(inplace=True, subset=['abstract'])
        data.drop_duplicates(inplace=True, subset=['title_standardized'])
        data.drop_duplicates(inplace=True, subset=['abstract_processed'])
        # data.dropna(axis=0, how='any')
        data.reset_index(inplace=True)
        data.drop(axis=1, labels=['index'], inplace=True)


        authors = {}
        for author in data.HDSI_author.unique():
            authors[author] = {
                2015 : list(),
                2016 : list(),
                2017 : list(),
                2018 : list(),
                2019 : list(),
                2020 : list(),
                2021 : list()
            }

        for i, row in data.iterrows():
            authors[row['HDSI_author']][row['year']].append(row['abstract_processed'])

        all_docs = []
        missing_author_years = {author : list() for author in data.HDSI_author.unique()}
        for author, author_dict in authors.items():
            for year, documents in author_dict.items():
                if len(documents) == 0:
                    missing_author_years[author].append(year)
                    continue
                all_docs.append(" ".join(documents))

        return data, authors, all_docs, missing_author_years

    
    dataframe, authors, all_docs, missing_author_years = data_cleaning(test_data)
    pickle.dump(dataframe, open("test/interim/dataframe.pkl", 'wb'))
    pickle.dump(authors, open("test/interim/authors.pkl", 'wb'))
    pickle.dump(all_docs, open("test/interim/all_docs.pkl", 'wb'))
    pickle.dump(missing_author_years, open("test/interim/missing_author_years.pkl", 'wb'))

    countVec = CountVectorizer()
    all_docs = pickle.load(open("test/interim/all_docs.pkl", 'rb'))
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()

    # save models to pkl file

    models = {}
    results = {}
    num_topics = [15]
    for num_components in num_topics:
        modeller = LatentDirichletAllocation(n_components=num_components, n_jobs=-1, random_state=123)
        result = modeller.fit_transform(counts)
        models[str(num_components)] = modeller
        results[str(num_components)] = result

    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
    display_topics(modeller, names, 15)
    # path = r'results/model_prediction/'
    # results.to_csv(path + 'time_author_topic.csv', index=False)

    # 'src/models/names.pkl'
    pickle.dump(names, open("test/processed/names.pkl", 'wb'))
    # 'src/models/10_15_20_25_30_models.pkl'
    pickle.dump(models, open("test/processed/models.pkl", 'wb'))
    # src/models/10_15_20_25_30_results.pkl
    pickle.dump(results, open("test/processed/results.pkl", 'wb'))
    # 'src/models/10_15_20_25_30_vectorizer.pkl'
    pickle.dump(countVec, open("test/processed/countvec.pkl", 'wb'))

    print("done")

if __name__ == "__main__":
    main(sys.argv[0])


    # ---- CODE THAT MAY BE USED DURING Q2 ----
    # data preprocssing
    # hdsi_faculty["abstract"].fillna(hdsi_faculty["title"], inplace=True) # if no abstract, replace w/ title of article
    # hdsi_faculty = hdsi_faculty[hdsi_faculty["year"] > 2014]
    # #combining all the documents into a list by author and year:

    # authors = {}
    # for author in hdsi_faculty['HDSI_author'].unique():
    #     authors[author] = {
    #         2015 : list(),
    #         2016 : list(),
    #         2017 : list(),
    #         2018 : list(),
    #         2019 : list(),
    #         2020 : list(),
    #         2021 : list()
    #     }
    
    # for i, row in hdsi_faculty.iterrows():
    #     authors[row['HDSI_author']][row['year']].append(row['abstract'])
    
    # corpus = []
    # for author, author_dict in authors.items():
    #     for year, documents in author_dict.items():
    #         corpus.append(" ".join(documents))

    # # text preprocessing on corpus

    # # stop loss words 
    # stop = set(stopwords.words('english'))

    # exclude = set(string.punctuation)

    # # lemmatization
    # lemma = WordNetLemmatizer()

    # # one function for all the steps:
    # def clean(doc):
         
    #     # convert text into lower case + split into words
    #     stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
         
    #     # remove any stop words present
    #     punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
         
    #     # remove punctuations + normalize the text
    #     normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
         
    #     return normalized

    # # clean data stored in a new list

    # clean_corpus = [clean(doc).split() for doc in corpus]

    # # covert text into numerical representation

    # # Converting text into numerical representation
    # cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    # print(cv_vectorizer)

    # # Array from Count Vectorizer 
    # cv_arr = cv_vectorizer.fit_transform(clean_corpus)

    # # Creating vocabulary array which will represent all the corpus 
    # vocab_cv = cv_vectorizer.get_feature_names()

    # # implementation of LDA:
         
    # # Create object for the LDA class 
    # # Inside this class LDA: define the components:
    # lda_model = LatentDirichletAllocation(n_components = 15, n_jobs=-1, random_state=123)

    # # fit transform on model on our count_vectorizer : running this will return our topics 
    # X_topics = lda_model.fit_transform(cv_arr)

    # # components_ gives us our topic distribution 
    # topic_words = lda_model.components_

    # n_top_words = 15

    # for i, topic_dist in enumerate(topic_words):
         
    #     # np.argsort to sorting an array or a list or the matrix acc to their values
    #     sorted_topic_dist = np.argsort(topic_dist)
         
    #     # Next, to view the actual words present in those indexes we can make the use of the vocab created earlier
    #     topic_words = np.array(vocab_cv)[sorted_topic_dist]
         
    #     # so using the sorted_topic_indexes we are extracting the words from the vocabulary
    #     # obtaining topics + words
    #     # this topic_words variable contains the Topics  as well as the respective words present in those Topics
    #     topic_words = topic_words[:-n_top_words:-1]
    #     print ("Topic", str(i+1), topic_words)

    # # To view what topics are assigned to the documents:

    # doc_topic = lda_model.transform(cv_arr) 

    # # column names
    # topicnames = ["Topic" + str(i) for i in range(15)]

    # # index names
    # docnames = ["Doc" + str(i) for i in range(len(corpus))]

    # # Make the pandas dataframe
    # df_document_topic = pd.DataFrame(doc_topic, columns=topicnames, index=docnames)

    # # Get dominant topic for each document
    # dominant_topic = np.argmax(df_document_topic.values, axis=1)
    # df_document_topic['dominant_topic'] = dominant_topic

    # df_document_topic['author'] = np.nan
    # df_document_topic['year'] = np.nan
    # df_document_topic.shape

    # year_paper_count = {}
    # for author in authors.keys():
    #     if author not in year_paper_count.keys():
    #         year_paper_count[author] = 0
    #     year_paper_count[author] += len(authors[author])

    # author_list = list(year_paper_count.keys())
    # for i in range(0, len(corpus), 7):
    #     df_document_topic.iloc[i:i+7, 16] = author_list[i//7]
    #     year = 2015
    #     for j in range(i, i+7):
    #         df_document_topic.iloc[j, 17] = str(year)
    #         year += 1
    # time_author_topic = df_document_topic

    # path = r'results/model_prediction/'
    # time_author_topic.to_csv(path + 'time_author_topic.csv', index=False)

