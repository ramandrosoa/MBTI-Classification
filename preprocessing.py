import spacy
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

nlp = spacy.load("en_core_web_sm")
sentiments = SentimentIntensityAnalyzer()

def word_counts(df):
    #Remove links
    def remove_links(text):
        return re.sub(r'http\S+|www\.\S+', '', text, flags = re.IGNORECASE)
    df['posts'] = df['posts'].apply(remove_links)
    
    df['word_counts'] = df['posts'].apply(lambda x : len(x.split()))
    return df

def avg_sentence_length(df):
    def extract(text):
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        sentences_count = len(sentences)
        return sentences_count
    df["sentence_count"] = df['posts'].apply(extract)
    df["avg_sentence_length"] = df["word_counts"]/df["sentence_count"]
    return df

def perc_long_words(df) :
    
    #Count Syllables
    def syllable_count(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        #Check the first character
        if word[0] in vowels:
            count +=1
        for index in range(1,len(word)):
        
        #Prevent counting multiple consecutive vowels as separate syllables
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith("es") or word.endswith("ed"):
            count-=1
        
        #Ensure it returns at least one syllable
        return max(1,count)
    
    
    #Count long words
    def count_long_words(post):
        long_word_count = 0
        for word in post.split():
            if syllable_count(word) >2:
                long_word_count += 1
        return long_word_count
    
    df['perc_longword'] = ((df['posts'].apply(count_long_words))/df['word_counts'])*100
    
    return df
        
distinct_entities = ['LOC', 'ORDINAL', 'PERSON',
                     'FAC', 'EVENT', 'CARDINAL',
                     'ORG', 'WORK_OF_ART', 'TIME',
                     'QUANTITY', 'NORP', 'PRODUCT',
                     'GPE', 'MONEY', 'LAW', 'DATE',
                     'PERCENT', 'LANGUAGE']


def preprocessing(text):
    
    if not isinstance(text, str): #Check if text is any datatype other than a string
        try:
            text = str(text) #attempt to convert text to string
        except:
            return "" #return empty string if conversion fails
        
    #tokenization 
    tokens = word_tokenize(text)

    #Remove stopwords NLTK but keep negation words
    mbti_stopwords = {'intp','intj','infp','infj',
                         'entp','entj','enfj','enfp',
                         'esfp','esfj','estp','estj',
                        'isfp','isfj','istp','istj',
                      
                     'INTP', 'INTJ','INFP', 'INFJ',
                     'ENTP','ENTJ','ENFJ','ENFP',
                     'ESFP','ESFJ','ESTP','ESTJ',
                      'ISFP','ISFJ','ISTP','ISTJ'}
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'never'}
    mystop_words = mbti_stopwords.union(stop_words)
    tokens  =  [word for word in tokens if word not in mystop_words]

    #Lemmetization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
  
    #Join tokens 
    text_str = ' '.join(tokens)

    return text_str

def sentiment_score(df):
    df['Positive'] = [sentiments.polarity_scores(i)['pos']
                     for i in df['preprocessed_text']]
    
    df['Negative'] = [sentiments.polarity_scores(i)['neg'] 
                     for i in df['preprocessed_text']]
    
    df['Neutral'] = [sentiments.polarity_scores(i)['neu']
                    for i in df['preprocessed_text']]
    
    return df

def convert_text_test(df, vectorizer, svd):
    
    #Vectorization
    text = df['preprocessed_text']
    text_vectorized = vectorizer.transform(text)
    
    #Dimensionality Reduction
    text_svd = svd.transform(text_vectorized)

    #Convert to df 
    text_df = pd.DataFrame(text_svd,
                           columns = [f'svd_feature_{i}'
                                       for i in range(text_svd.shape[1])],
                            index = text.index)
    
    #Drop columns
    df = df.drop(columns = ['preprocessed_text'],
                 axis = 1)
    
    df = pd.concat([df,text_df], axis = 1)

    return df

def test_prep (X, vectorizer, svd , scaler):
    
    #Post length and Readability 
    X = word_counts(X)
    X = avg_sentence_length(X)
    X = perc_long_words(X)
    X['gfi_score'] = 0.4 * (X['avg_sentence_length'] + X["perc_longword"])
    
    #Named entity recognition
    def ner(text) :
            doc = nlp(text)
            entities = defaultdict(list)
            for ent in doc.ents : 
                entities[ent.label_].append(ent.text)
            return dict(entities)
    X['named_entities_recognition'] = X['posts'].apply(ner)
    
    for entity in distinct_entities:
        X[entity] = 0
    for idx, row in X.iterrows():
        entity_dict = row['named_entities_recognition']
        for entity in distinct_entities:
            if entity in entity_dict:
                X.at[idx, entity] = len(entity_dict[entity])
    

    #Sentiment Analysis 
    X["preprocessed_text"] = X["posts"].apply(preprocessing)
    X = sentiment_score(X)
    X['polarity'] = (X['Positive']-X['Negative'])/((X['Positive']+X['Negative'])*100)
    
    X = X.drop(columns = ['posts', 'named_entities_recognition', 
                         'Positive', 'Negative', 'Neutral', 
                         'perc_longword', 'avg_sentence_length', 'sentence_count'], 
              axis = 1)
    
    #Vectorization and Transformation 
    X = convert_text_test(X, vectorizer, svd)
    
    
    #Scaling
    X_test_scaled = scaler.transform(X)
    

    return X_test_scaled
