from __future__ import print_function
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from time import time

import sys, os, re, pickle, gzip, json, nltk, math, csv, logging, gensim, operator
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from itertools import combinations

from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier




"""
Section 1. Read tweets for entities and then select cause-relevant tweets as training data to label
"""

def get_brand_info(gg_file,name_field,cause_field):
    """
    Read info from GoodGuide file, get brands and attribute(eco,health) scores.
    Parameters:
        gg_file: goodguide score file
        name_field: for health brands, name_field = "screen_name"; for eco brands, name_field = "twitter"
        cause_field: for health, cause_field = "health score"; for eco brands, cause_field = "TGS"
    Return:
        brand_score_dict: dict[brand_name] = brand's goodguide score
        brand_sector: dict[brand_name] = sector
        brand_nameid: dict[brand_name] = brand's identification name
    """
    brand_score_dict = {}
    brand_sector = {}
    brand_nameid = {}
    with open(gg_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row[name_field] and row[name_field] not in brand_score_dict):
                if(row[cause_field] != "N/A"):
                    brand_score_dict[row[name_field]] = row[cause_field]
                    brand_sector[row[name_field]] = row['sector']
                    brand_nameid[row[name_field]] = row['name']
                
    csvfile.close()
    print("Get %d brands with %s" % (len(brand_score_dict),cause_field))
    if(cause_field == "TGS"):
        return brand_score_dict,brand_sector,brand_nameid
    elif(cause_field == "health score"):
        return brand_score_dict,brand_sector


def get_healthbrand_nameid(eco_gg_file):
    """
    Parameters:
        Read brands in big goodguide eco score file, and keep those in food and personal care sectors
    Return:
        brand_nameid: dict[brand_screen_name] = brand's real name
    """
    
    brand_nameid = {}
    with open(eco_gg_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row['twitter'] and row['twitter'] not in brand_nameid):
                if(row['sector'] == "Food" or row['sector'] == "Personal Care"):
                    brand_nameid[row['twitter']] = row['name']
                    
    return brand_nameid
    
    
def read_brand_tweets(tweetfile, entitylist,cause):
    """
    Parameters:
        tweetfile: tweet data file
        entitylist: a list of brands or congress members
        cause: eco or health
    Return:
        dict[entity]  = [A list of the entity's tweets]
    """
    lineno = 0
    nread = 0
    nentity = 0
    entity_tweets_dict = {}

    for line in gzip.open(tweetfile, 'rt'):
        js = json.loads(line)
        name = js['user']['screen_name'].lower()
        if name in entitylist:
            # This following code removes line breaks inside a tweet message.
            clean_text = re.sub('\n', ' ', re.sub('\r',' ',re.sub('\t',' ',js['text'].strip())))
            if name in entity_tweets_dict:
                entity_tweets_dict[name].append(clean_text) 
            else:
                nentity += 1
                entity_tweets_dict[name] = []
                entity_tweets_dict[name].append(clean_text) 
            nread += 1
        lineno += 1
        if lineno % 500000 == 0:
            print('read %d lines' % lineno)
    print('Collected %d tweets for %d %s brands in total.' % (nread,nentity,cause))
    return entity_tweets_dict


def read_moc_tweets(mocfile):
    """
    Parameters:
        moc: short for member of Congress
        mocfile: datafile for mocs
    Return:
        Read tweets for all congress members.
        moc_tweets_dict: dict[moc] = [A list of his/her tweets]
        moc_nameid_dict: dict[moc_screen_name] = moc_real_name 
        moc_party_dict: dict[moc_screen_name] = moc's party
        moc_state_dict: dict[moc_screen_name] = moc's state
    """
    nline = 0
    nread = 0
    screen_name_set = []
    moc_rating = {}
    moc_nameid_dict = {}
    moc_party_dict = {}
    moc_state_dict = {}
    MOC_tweets_dict = {}

    for line in open(mocfile, 'rt'):
        js = json.loads(line)
        screen_name = js['screen_name']
        real_name = js['mocName']
        rating = float(js['mocLifetimeRating'].strip('%')) / 100
        party = js['mocParty']
        state = js['mocState']
        if screen_name not in MOC_tweets_dict:
            moc_nameid_dict[screen_name] = real_name
            moc_rating[screen_name] = rating
            moc_party_dict[screen_name] = party
            moc_state_dict[screen_name] = state
            
            nline += 1
            if(nline % 100 == 0):
                print("Read tweets for %d congress members" % nline)
            
            MOC_tweets_dict[screen_name]=[]
            for tweet in js['tweets']:
                nread += 1
                tweet_clean = re.sub('\n', ' ', re.sub('\r',' ',re.sub('\t',' ',tweet['text'].strip())))
                MOC_tweets_dict[screen_name].append(tweet_clean)
        else:
            print("Reapeted congress member: %s" % screen_name)
            
    print('Collected %d tweets for %d congress members in total.' % (nread,nline))

    return moc_nameid_dict, moc_rating, moc_party_dict, moc_state_dict, MOC_tweets_dict


def tw_tokenize(tweet):
    """
    Return a list of words in a tweet message.
    """
    return re.findall('\w+',         
                              re.sub('\s+', ' ', 
                               re.sub('\d+',' ',
                               re.sub('RT', ' ', 
                               re.sub('http\S+', ' ',
                               re.sub('@\S+', ' ', tweet.lower())
                                                              ))).strip()))


def dedup_tweets(entity_tweets_dict,cause):
    """
    Parameters:
        entity_tweets_dict: dict[brand_name] = [a list of this brand's tweets]
    Return:
        Remove deduped tweets for each brand.
        entity_tweetsID_dict[brandname] = [A list of tweets' IDs]
        ID_tweet[tweet_ID] = tweet text
    """
    entity_tweetsID_dict = {}
    ID_tweet = {}
    ID = 0
    nentity = 0
    for entity in entity_tweets_dict:
        nentity += 1
        if(nentity % 100 ==0):
            print("processed %d entities" % (nentity))
        seen = set()
        dedupedID = []
        for tweet in entity_tweets_dict[entity]:
            text = ' '.join(tw_tokenize(tweet))
            if text not in seen:
                seen.add(text)
                ID_tweet[ID] = tweet#remove \n inside a tweet.
                dedupedID.append(ID)
                ID += 1
        entity_tweetsID_dict[entity] = dedupedID
    
    print('%d non-duplicate tweets for %d %s brands' % (len(ID_tweet), len(entity_tweetsID_dict), cause))
    return entity_tweetsID_dict,ID_tweet


def load_GoogleNews_w2v(w2v_GN_file):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("Start loading GoogleNews word2vec model.")
    GN_model = gensim.models.Word2Vec()
    GN_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_GN_file,binary = True) 
    #GN_model = gensim.models.Word2Vec.load_word2vec_format(w2v_GN_file,binary = True) 
    print("The vocabulary size is: "+str(len(GN_model.vocab)))
    return GN_model


def get_wds_inW2V(GN_model,tweet):
    """
    Parameters: 
        tweet: tweet message
    Return:
        A list of tokens that has corresponding vector representation in GoogleNews Word2Vec model.
    """
    tw_tokens = tw_tokenize(tweet)
    W2V_tokens = []
    for token in tw_tokens:
        if((token in GN_model.vocab) and (token.lower() not in stopwords.words('english'))):
            W2V_tokens.append(token)
    return W2V_tokens

def score_tweet_by_relevance(ID_tweet,GN_model,keywords):
    print("Note: This function takes some time to run. Please run for once, and save results to file.")
    """
    Parameters:
        ID_tweet: dict[tweetID] = tweet text
        keywords: list of cause keywords
    Return:
        Calculate each tweet's similarity with keywords by word2vec model.
        ID_twScore: dict[tweetID] = similarity score
    Note:
        This function takes about 60 minutes to run. Only run for once, and save results to file.
    """
    ID_twScore = {}
    ntw = 0
    for ID in ID_tweet:
        ntw += 1
        if(ntw %100000 == 0):
            print("processed %d tweets" % ntw)
        
        tw_w2v_wds = get_wds_inW2V(GN_model,ID_tweet[ID])
        if(len(tw_w2v_wds) > 0):
            ID_twScore[ID] = GN_model.n_similarity(tw_w2v_wds,keywords)
        else:
            ID_twScore[ID] = 0.0
    
    return ID_twScore


def sort_tweet_by_score(ID_twScore, entity_twID_dict):
    """
    Parameters:
        ID_twScore: dict[tweetID] = cause relevant score
        entity_twID_dict[entity] = [A list of tweet's IDs]
    Return:
        For each entity, sort all tweets by cause-relevant scores.
        entity_twID_score: dict[entity] = [A list of tuples(tweetID, relevant score) sorted by cause relevant score]
    """
    entity_twID_score = {}
    for entity in entity_twID_dict:
        twID_list = entity_twID_dict[entity]
        twID_score_dict = {twID:ID_twScore[twID] for twID in twID_list}
        sorted_twID_score = sorted(twID_score_dict.items(),key=operator.itemgetter(1),reverse=True)
        entity_twID_score[entity] = sorted_twID_score #list of tuples
    return entity_twID_score


def select_entity_topn(filename,entity_twIDScore,ID_tweet,topn):
    """
    Parameters:
        filename: write each entity's tweets' cause relevant scores to file.
        entity_twIDScore: dict[entity] = [A list of tuples(tweetID, relevant score) sorted by cause relevant score]
        ID_tweet: dict[tweetID] = tweet text.
        topn: number of high relevant tweets to extract.
    """
    with open(filename,"w") as csvfile:
        fieldnames = ['brand', 'tweetID','tweet_score','tweet_text']
        for entity in entity_twIDScore:
            ID_score_list = entity_twIDScore[entity]
            mywriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for i in range(min(len(ID_score_list),topn)):
                mywriter.writerow({'brand': entity, 'tweetID': ID_score_list[i][0], 'tweet_score': "%.3f" % ID_score_list[i][1],
                             'tweet_text':ID_tweet[ID_score_list[i][0]]})
    csvfile.close()

"""
Section2: read training instances and explore basic information.
"""
def data_for_sup_clf(labeled_file,entity):
    """
    Parameters:
        labeled_file: The file containing labeled data.
    Return:
        labels 0 1 are non-support class (negative), labels 2 3 are support class(positive)
        sup_entity_list: a list of entities
        sup_tweet_list: a list of tweets
        sup_label_list: a list of labels for tweets
    """
    sup_tweet_list = []
    sup_entity_list = []
    sup_label_list = []
    
    with open(labeled_file,'r') as infile:
        myreader = csv.DictReader(infile)
        n_row = 0
        for row in myreader:
            if(row['label']=='0' or row['label']=='1'):
                sup_label_list.append(0)
            elif(row['label']=='2' or row['label']=='3'):
                sup_label_list.append(1)
            sup_tweet_list.append(str(row['tweet']))
            sup_entity_list.append(str(row[entity]))        
    infile.close()
    sup_ct = Counter()
    sup_ct.update(sup_label_list)
    print("Read %d positive instances and %d negative instances for support classification" % (sup_ct[1],sup_ct[0]))
    return sup_entity_list, sup_tweet_list, sup_label_list


def data_for_commit_clf(labeled_file,entity):
    """
    Parameters:
        labeled_file: The file containing labeled data.
    Return:
        label-2 is low-commitment class(negative), label-3 is high-commitment class(positive)
        comt_entity_list: a list of entities
        comt_tweet_list: a list of tweets
        comt_label_list: a list of labels for tweets
    """
    comt_tweet_list = []
    comt_entity_list = []
    comt_label_list = []
    
    with open(labeled_file,'r') as infile:
        myreader = csv.DictReader(infile)
        n_row = 0
        for row in myreader:
            if(row['label'] == '2'):
                comt_label_list.append(0)
                comt_tweet_list.append(str(row['tweet']))
                comt_entity_list.append(str(row[entity]))
            elif(row['label'] == '3'):
                comt_label_list.append(1)
                comt_tweet_list.append(str(row['tweet']))
                comt_entity_list.append(str(row[entity]))
            
    infile.close()
    comt_ct = Counter()
    comt_ct.update(comt_label_list)
    print("Read %d positive instances and %d negative instances for commitment classification" % (comt_ct[1],comt_ct[0]))
    return comt_entity_list, comt_tweet_list, comt_label_list


def tw_tokenize_with_features(tweet):
    """
    Parameters:
        tweet: a tweet message
    Return:
        A list of tokens, tokenize tweet by space and change special characters into readable signs.
    """
    return re.findall('[a-zA-Z0-9_\']+',
                              re.sub('\s+',' ',
                              re.sub(' \d+',' _NUMBER_',
                              re.sub('#(\S+)',r'_HASHTAG_\1',
                              re.sub('http\S+','_URL_',
                              re.sub('@(\S+)',r'_MENTION_\1',tweet.strip().lower()))))))


def get_freq_terms(tweet_list,label_list):
    """
    Parameters:
        tweet_list: a list of tweets
        label_list: a list of corresponding labels
    Return:
        neg_terms: a counter for words in negative class
        pos_terms: a counter for words in positive class
    """
    
    neg_terms = Counter()
    pos_terms = Counter()
    
    for i in range(len(tweet_list)):
        tokens = set(tw_tokenize_with_features(tweet_list[i])).difference(stopwords.words('english'))
        if(label_list[i] == 0):
            neg_terms.update(tokens)
        elif(label_list[i] == 1):
            pos_terms.update(tokens)
        
    return neg_terms, pos_terms


"""
Section3: feature engineering: linguistic cues and word2vec vectors
"""
def construct_feature_matrix(text_list, stopwords = 'english',my_min_df=1, my_max_df=1.0, my_ngram=(1,1), mytokenizer=tw_tokenize_with_features,flag = "ct"):
    vectorizer = CountVectorizer(stop_words = stopwords, min_df=my_min_df, max_df=my_max_df,ngram_range=my_ngram,tokenizer = mytokenizer)
    X_ct_matrix = vectorizer.fit_transform(text_list)
    if(flag == 'ct'):
        return vectorizer,X_ct_matrix.toarray()
    elif(flag == 'tfidf'):
        transformer = TfidfTransformer()
        X_tfidf_matrix = transformer.fit_transform(X_ct_matrix)
        return vectorizer,X_tfidf_matrix.toarray()



def construct_feature_matrix_formoc(text_list,mytokenizer = tw_tokenize_with_features,flag = "ct",stop_words_flag='english'):
    "Construct feature matrix for text/word features."
    "Need to remain stop words in this classification task."
    vectorizer = CountVectorizer(min_df=3, max_df=0.75,tokenizer = mytokenizer)#stop_words = 'english'
    X_ct_matrix = vectorizer.fit_transform(text_list)
    if(flag == 'ct'):
        return vectorizer,X_ct_matrix.toarray()
    elif(flag == 'tfidf'):
        transformer = TfidfTransformer()
        X_tfidf_matrix = transformer.fit_transform(X_ct_matrix)
        return vectorizer,X_tfidf_matrix.toarray()

def mark_polarity(my_tweet_list,to_wd):
    r_neg = re.compile("( not )|( no )|( don't )|( doesn't )|( didn't )|( wouldn't )")
    tweet_addpolarity = []
    for tweet in my_tweet_list:
        if(re.search(r_neg,tweet)):
            if(to_wd == 0):#not add to word, but only regard as a separate feature.
                pola_text = re.sub(r_neg, " _NEG_ ",tweet)
            elif(to_wd == 1):#add it to the right word.
                pola_text = re.sub(r_neg, " _NEG_",tweet)
            elif(to_wd == 2):#add it to all words in the text.
                pola_text = re.sub('([a-zA-Z0-9_\']+)',r'_NEG_\1',tweet)
        else:
            pola_text = tweet
        tweet_addpolarity.append(pola_text)    
    return tweet_addpolarity


def mark_pronouns(my_tweet_list,binary):
    tweet_addPron = []
    for tweet in my_tweet_list:
        fp = len(re.findall(r'\b(me|mine|i|our|ours|we|my|myself|us)\b', tweet, re.IGNORECASE))
        sp = len(re.findall(r'\b(you|your|yours|y\'all|yall|yourself)\b', tweet, re.IGNORECASE))
        tp = len(re.findall(r'\b(they|their|them|he|she|her|his|theirs|themselves|herself|himself)\b', tweet, re.IGNORECASE))
        if not binary:
            fp = max(1, fp)
            sp = max(1, sp)
            tp = max(1, tp)
        
        tweet_addPron.append(tweet + ' first__person' * fp + ' second__person' * sp + ' third__person' * tp)
    return tweet_addPron


def read_eco_terms(eco_file):
    """
    Read eco terms/expressions from txt 
    """
    eco_terms = set(t.strip() for t in open(eco_file))
    eco_terms -= set(['organics?', 'gmo'])
    return re.compile(r'\b(%s)\b' % '|'.join(sorted(eco_terms)), re.IGNORECASE)


def mark_context(my_tweet_list,keywords_regex):
    """
    Mark context of cause keywords. E.g., to represent the terms that occur just before/after an eco-term.
    """
    
    tweet_addContx = []
    for tweet in my_tweet_list:
        toadd = ''
        text = ' '.join(tw_tokenize_with_features(tweet))
        for m in keywords_regex.finditer(text):
            left = re.findall('\w+', text[:m.start()])
            toadd += ' left_context_%s' % left[-1].lower() if len(left) > 0 else ''
            right = re.findall('\w+', text[m.end():])
            toadd += ' right_context_%s' % right[0].lower() if len(right) > 0 else ''
        
        tweet_addContx.append(text + toadd)    
    return tweet_addContx


def remove_keywords(my_tweet_list,keywords_regex):
    tweet_rmTerm = []
    tweet_contx_list = mark_context(my_tweet_list, keywords_regex)
    for tweet_contx in tweet_contx_list:
        tweet_rmTerm.append(keywords_regex.sub(' ',tweet_contx))
    return tweet_rmTerm


def selfmention(entity_list,entity_nameid,my_tweet_list,count):
    """
    Add _self_ as a feature to indicate if a tweet mentions the entity itself.
    Parameters:
        entity_list: a list of entities' screennames
        entity_nameid: dict[entity's screen name] = entity's real name
        my_tweet_list: a list of tweets
    Return:
        A list of tweets that are marked with _self_ feature.
    """
    
    tweet_markself = []
    for i in range(len(my_tweet_list)):
        name_parts = re.findall("\w+",re.sub("_"," ",entity_list[i]))
        self_marks = [str(entity_list[i]),str(entity_nameid[entity_list[i]]),"".join(name_parts)]
        
        text = "".join(re.findall("\w+",my_tweet_list[i]))
        
        flag = False
        for mark in self_marks:
            if(re.search(mark,text,flags = re.I)):
                flag=True
                break
            
        if(flag == True):
            if(count == "once"):
                tweet_markself.append(my_tweet_list[i]+" _SELF_")
            elif(count == "all"):
                wd_list = my_tweet_list[i].split(" ")
                new_text = ""
                for wd in wd_list:
                    new_text += "_SELF_"+wd+" "
                tweet_markself.append(new_text)
        elif(flag == False):
            tweet_markself.append(my_tweet_list[i])
        
    return tweet_markself

def mark_pos(my_tweet_list):
    """
    Returns:
        tweet_taglist: a list of pos tags.
        tweet_and_tag_list: a list of words and their corresponding pos tags
    """
    n = 0
    tweet_taglist = []
    tweet_and_tag_list = []
    
    for tweet in my_tweet_list:
        if(len(tweet)>0):
            word_list = tw_tokenize_with_features(tweet)
            tuple_list = nltk.pos_tag(word_list)
            tags = ""
            wdtags = ""
            for item in tuple_list:
                tags += item[1]+" "
                wdtags += item[0]+" "+item[1]+" "
            tweet_taglist.append(str(tags))
            tweet_and_tag_list.append(str(wdtags))
    
    return tweet_taglist,tweet_and_tag_list


def construct_tw_vector(my_tweet_list,GN_model):
    """
    Returns:
        tw_vector: vector representation for tweet
    """
    tw_vector = []
    tweet_matchwd_list = []
    for tw in my_tweet_list:
        tweet_matchwd_list.append(get_wds_inW2V(GN_model,tw))
    for match_wd_list in tweet_matchwd_list:
        if(len(match_wd_list) > 0):
            w2v_list = []
            for wd in match_wd_list:
                w2v_list.append(GN_model[wd])
            tw_vector.append(np.array([sum(e)/len(w2v_list) for e in zip(*w2v_list)]))
        else:
            tw_vector.append(np.array([0.0]*300))
    return np.array(tw_vector)


def calculate_tw_w2v_score(my_tweet_list,GN_model,cause_keywords):
    """
    Returns:
        tweet_score_list: a list of tweets' cause-relevance scores.
    """
    tweet_score_list = []
    tweet_matchwd_list = []
    for tw in my_tweet_list:
        tweet_matchwd_list.append(get_wds_inW2V(GN_model,tw))
    for match_wd_list in tweet_matchwd_list:
        if(len(match_wd_list) > 0):
            tweet_score_list.append([float("%.3f" % GN_model.n_similarity(match_wd_list,cause_keywords))]) 
        else:
            tweet_score_list.append([0.0])
    return np.array(tweet_score_list)


def rank_match_words(my_tweet_list, GN_model, cause_keywords):
    """
    Returns:
        tweet_rankedwd_list: list of list, each sublist contains tuples ranked by words' cause-relevance scores. 
                            E.g.,[('fish', '0.377'), ('scent', '0.188')]
    """
    tweet_matchwd_list = []
    for tw in my_tweet_list:
        tweet_matchwd_list.append(get_wds_inW2V(GN_model,tw))
    tweet_rankedwd_list = []
    for this_wd_list in tweet_matchwd_list:
        wd_score_dict = {}
        for wd in this_wd_list:
            wd_score_dict[wd] = "%.3f" % GN_model.n_similarity([wd], cause_keywords)
        sorted_wd_score_dict = sorted(wd_score_dict.items(), key=operator.itemgetter(1),reverse=True)
        tweet_rankedwd_list.append(sorted_wd_score_dict)
    return tweet_rankedwd_list


def get_topn_words(tweet_rankedwd_list,n):
    """
    Parameters:
        tweet_rankedwd_list: list of list, each sublist contains tuples ranked by words' cause-relevance scores. 
    Returns:
        tweet_topnwd: a list of tweets' top-n words.
        tweet_topnwd_scores: a list of tweets' top-n words' cause relevance scores.
    """
    tweet_topnwd = []
    tweet_topnwd_scores = []
    for this_rankedwd_list in tweet_rankedwd_list:
        topnwd = ""
        topnwd_score = [0.0]*n
        for i in range(min(n,len(this_rankedwd_list))):
            topnwd += this_rankedwd_list[i][0]+" "
            topnwd_score[i] = float(this_rankedwd_list[i][1])
        tweet_topnwd.append(topnwd)
        tweet_topnwd_scores.append(topnwd_score)
        
    return tweet_topnwd, np.array(tweet_topnwd_scores)


def get_topn_vectors(tweet_rankedwd_list,GN_model,n):
    """
    Parameters:
        tweet_rankedwd_list: list of list, each sublist contains tuples ranked by words' cause-relevance scores. 
    Returns:
        tweet_topnwd_vectors: vector representation for each tweet's topn words
    """
    tweet_topnwd_vectors = []
    ct = 0
    for this_rankedwd_list in tweet_rankedwd_list:
        ct += 1
        topnwd_vectors = []
        for i in range(min(n,len(this_rankedwd_list))):
            topnwd_vectors.append(GN_model[this_rankedwd_list[i][0]])
    
        if(len(topnwd_vectors) == 0):
            tweet_topnwd_vectors.append([0.0]*300)
        else:
            tweet_topnwd_vectors.append([sum(e)/len(topnwd_vectors) for e in zip(*topnwd_vectors)])
        
        
    return np.array(tweet_topnwd_vectors)


def get_context_contri(GN_model,wd,i,tw_wd_list):
    """
    Parameters:
        tw_wd_list: a list of tweets's words
        wd: current word
        i: current word's index in tw_wd_list
    Returns:
        left_contri: current word's left word's contribution score
        right_contri: current word's right word's contribution score
    """
    if(i == 0):
        left_wd = ""
    else:
        left_wd = tw_wd_list[i-1]
    if(left_wd in GN_model.vocab):
        left_contri = float("%.3f" % GN_model.similarity(wd,left_wd))
    else:
        left_contri = 0.0
    if(i == len(tw_wd_list)-1):
        right_wd = ""
    else:
        right_wd = tw_wd_list[i+1]
    if(right_wd in GN_model.vocab):
        right_contri = float("%.3f" % GN_model.similarity(wd,right_wd))
    else:
        right_contri = 0.0
    
    return left_contri,right_contri


def get_topic_words(my_tweet_list,GN_model,cause_keywords,threshold):
    """
    Parameters:
        cause_keywords: a list of cause's keywords
        threshold: if a word's cause relevance score > 0.3, then take it as a cause word
    Returns:
        tweet_topicwd_list: a list of tweets' topic words
        tweet_topicwd_tp_list: list of dict, 
                                dict[word] = [word's cause-relevance score, left word's contribution, right word's contribution]
    """
    tweet_topicwd_list = []
    tweet_topicwd_tp_list = []
    for tweet in my_tweet_list:
        tw_wd_list = []
        tw_wd_list = tw_tokenize(tweet)
        topic_wds = ""
        wd_score_dict = {}
        for i in range(len(tw_wd_list)):
            wd = tw_wd_list[i]
            if((wd in GN_model.vocab) and (wd.lower() not in stopwords.words('english'))):
                wd_score = float("%.3f" % GN_model.n_similarity([wd],cause_keywords))
                if(wd_score >= threshold):
                    topic_wds += wd+" "
                    left_contri,right_contri = get_context_contri(GN_model,wd,i,tw_wd_list)
                    wd_score_dict[wd] = [wd_score, left_contri, right_contri]
        if(len(wd_score_dict) == 0):
            wd_score_dict["NULL"] = [0.0,0.0,0.0]
        tweet_topicwd_list.append(topic_wds)
        tweet_topicwd_tp_list.append(wd_score_dict)
                
    return tweet_topicwd_list,tweet_topicwd_tp_list


def get_topicwd_vec(GN_model,tweet_topicwd_tp_list):
    """
    Returns:
        vector representation of each tweet's topic words
    """
    topicwd_dim_list = []
    for item in tweet_topicwd_tp_list:
        topic_vec = []
        for key in item:
            topic_vec.append(GN_model[key])
        topicwd_dim_list.append([sum(e)/len(topic_vec) for e in zip(*topic_vec)])
    return np.array(topicwd_dim_list)


def sep_topic_features(tweet_topicwd_tp_list):
    """
    Returns:
        topic_wd_ct: number of topic words in each tweet
        topic_wd_relevance: a list of each tweet's topic words' cause-relevance scores
        topic_wd_leftcontri: a list of each tweet's topic words' left content contribution scores
        topic_wd_rightcontri: a list of each tweet's topic words' right content contribution scores
        
    """
    topic_wd_ct = []
    topic_wd_relevance = []
    topic_wd_leftcontri = []
    topic_wd_rightcontri = []
    for item in tweet_topicwd_tp_list:
        topic_wd_ct.append([len(item)])
        tmp_leftcontri = []
        tmp_rightcontri = []
        tmp_relevance = []
        for key in item:
            tmp_relevance.append(item[key][0])
            tmp_leftcontri.append(item[key][1])
            tmp_rightcontri.append(item[key][2])
        topic_wd_relevance.append(tmp_relevance)
        topic_wd_leftcontri.append(tmp_leftcontri)
        topic_wd_rightcontri.append(tmp_rightcontri)
    return np.array(topic_wd_ct),np.array(topic_wd_relevance),np.array(topic_wd_leftcontri),np.array(topic_wd_rightcontri)


def get_topicwd_score_sum(W2V_topicwd_score):
    """
    Parameters:
        W2V_topicwd_score: a list tweet's topic words' cause-relevance scores
    Returns:
        W2V_topicwd_sum: a list of summation of tweet's topic words' cause-relevance scores
    """
    W2V_topicwd_sum = []
    for i in range(len(W2V_topicwd_score)):
        W2V_topicwd_sum.append([np.sum(W2V_topicwd_score[i])])
    return np.array(W2V_topicwd_sum)


def get_contri_sum(tweet_topicwd_tp_list):
    """
    Parameters:
        tweet_topicwd_tp_list: list of dict, 
                                dict[word] = [word's cause-relevance score, left word's contribution, right word's contribution]
    Returns:
        contri_score_list: a list of [topic_word_relevance_score + left_contribution + right_contribution, ......]
    """
    contri_score_list = []
    for item in tweet_topicwd_tp_list:
        sorted_relevance = sorted(item.items(), key = operator.itemgetter(1), reverse = True)#no need to sort
        tmp_list = [0.0]*3
        for i in range(min(3,len(sorted_relevance))):
            tmp_list[i] = np.sum(sorted_relevance[i][1])
        contri_score_list.append(tmp_list)
    return np.array(contri_score_list)


def do_grid_search(pipeline,parameters,data,label,score):
    
    grid_search = GridSearchCV(pipeline, parameters, scoring=score, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data,label)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    


def construct_linguistic_matrix(tweet_list,eco_terms,keywords,entity_list,entity_nameid,mystopwords='english',my_maxdf=1.0,my_mindf=1):
    #bag-of-words
    BOW_vectorizer, BOW_tw_matrix = construct_feature_matrix(tweet_list,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + polarity
    BOW_addpola = mark_polarity(tweet_list,to_wd=0)
    BOW_neg_vectorizer, BOW_neg_matrix = construct_feature_matrix(BOW_addpola,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + pronoun
    BOW_addPron_tweet = mark_pronouns(tweet_list, binary = False)
    pron_vectorizer,BOW_pron_matrix = construct_feature_matrix(BOW_addPron_tweet,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + keywords' context
    BOW_addCont_tweet = mark_context(tweet_list,eco_terms)
    cont_vectorizer,BOW_cont_matrix = construct_feature_matrix(BOW_addCont_tweet,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + remove causekeywords
    BOW_rmTopic_tweet = remove_keywords(tweet_list, eco_terms)
    rmeco_vectorizer,BOW_rmTopic_matrix = construct_feature_matrix(BOW_rmTopic_tweet,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + self_mention
    BOW_self_tweet_once = selfmention(entity_list,entity_nameid,tweet_list, count="once")
    mention1_vectorizer,BOW_mentionOnce_matrix = construct_feature_matrix(BOW_self_tweet_once,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)
    
    #bag-of-words + self_mention to all words
    BOW_self_tweet_all = selfmention(entity_list,entity_nameid,tweet_list, count="all")
    mentionAll_vectorizer,BOW_mentionAll_matrix = construct_feature_matrix(BOW_self_tweet_all,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + self_mention + pronoun
    BOW_self_pron_tweet = selfmention(entity_list,entity_nameid,BOW_addPron_tweet, count="once")
    mention1pron_vectorizer,BOW_mentionOncePron_matrix = construct_feature_matrix(BOW_self_pron_tweet,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)

    #bag-of-words + self_mention to all words + pronoun
    BOW_selfall_pron_tweet = selfmention(entity_list,entity_nameid,BOW_addPron_tweet, count="all")
    mentionAllpron_vectorizer,BOW_mentionAllPron_matrix = construct_feature_matrix(BOW_selfall_pron_tweet,stopwords=mystopwords,my_max_df=my_maxdf,my_min_df=my_mindf)
    
    Linguistic_matrix = [BOW_tw_matrix,BOW_neg_matrix,BOW_pron_matrix,BOW_cont_matrix,BOW_rmTopic_matrix,BOW_mentionOnce_matrix,BOW_mentionAll_matrix,BOW_mentionOncePron_matrix,BOW_mentionAllPron_matrix]
    Linguistic_vectorizer = [BOW_vectorizer,BOW_neg_vectorizer,pron_vectorizer,cont_vectorizer,rmeco_vectorizer,mention1_vectorizer,mentionAll_vectorizer,mention1pron_vectorizer,mentionAllpron_vectorizer]
    
    return Linguistic_matrix,Linguistic_vectorizer


def eva_bow_feature(Lingu_feature_dict,target_label_list,clf,mycv,score_func):
    feature_score={}
    for feature in Lingu_feature_dict.keys():
        feature_score[feature] = np.mean(cross_val_score(clf, Lingu_feature_dict[feature],target_label_list,cv=mycv,scoring=score_func))
    sorted_feature_score = sorted(feature_score.items(),key=operator.itemgetter(1),reverse=True)
    for item in sorted_feature_score:    
        print("%s\t%.3f" % (item[0],item[1]))



def construct_w2v_matrix(tweet_list,cause_keywords,GN_model,my_maxdf=1.0,my_mindf=1):
    W2V_tw_score = calculate_tw_w2v_score(tweet_list,GN_model,cause_keywords)
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)

    tweet_rankedwd_list = rank_match_words(tweet_list, GN_model, cause_keywords)
    W2V_topnwd, W2V_topnwd_scores = get_topn_words(tweet_rankedwd_list,n=3)
    topn_vectorizer, W2V_topnwd_matrix = construct_feature_matrix(W2V_topnwd,my_max_df=my_maxdf,my_min_df=my_mindf)

    W2V_topnwd_vectors = get_topn_vectors(tweet_rankedwd_list,GN_model,n=3)

    W2V_topicwd_list,tweet_topicwd_tp_list = get_topic_words(tweet_list,GN_model,cause_keywords,threshold = 0.30)
    topic_vectorizer,Topicwd_matrix = construct_feature_matrix(W2V_topicwd_list,my_max_df=my_maxdf,my_min_df=my_mindf)

    W2V_topicwd_ct,W2V_topicwd_score,W2V_topicwd_leftcontri,W2V_topicwd_rightcontri = sep_topic_features(tweet_topicwd_tp_list)

    W2V_topicwd_sum = get_topicwd_score_sum(W2V_topicwd_score)

    W2V_contri_score = get_contri_sum(tweet_topicwd_tp_list)

    W2V_topicwd_vectors = get_topicwd_vec(GN_model,tweet_topicwd_tp_list)
    
    w2v_matrix = [W2V_tw_score,W2V_tw_vector,W2V_topnwd_scores,W2V_topnwd_matrix,W2V_topnwd_vectors,Topicwd_matrix,W2V_topicwd_ct,W2V_topicwd_score,W2V_topicwd_leftcontri,W2V_topicwd_rightcontri,W2V_topicwd_sum,W2V_contri_score,W2V_topicwd_vectors]
    w2v_vectorizer = [topn_vectorizer,topic_vectorizer]
    
    return w2v_matrix,w2v_vectorizer


def eva_w2v_feature(W2V_feature_dict,target_label_list,clf,mycv,score_func):
    feature_score={}
    for feature in W2V_feature_dict.keys():
        feature_score[feature] = np.mean(cross_val_score(clf, W2V_feature_dict[feature],target_label_list,cv=mycv,scoring=score_func))
    sorted_feature_score = sorted(feature_score.items(),key=operator.itemgetter(1),reverse=True)
    for item in sorted_feature_score:    
        print("%s\t%.3f" % (item[0],item[1]))


def eva_comb_feature(feature_names,feature_dict,target_label_list,clf,mycv,score_func):
    """Find best feature combinations.
    This function takes about 20 minutes to run."""
    feature_accuracy_dict = {}
    #for n in range(1,len(feature_names)+1): # too many combinations
    for n in range(1,6): # only to check at most 5 combinations
        print("Combine %d features." % n)
        for indices in combinations(range(len(feature_names)),n):
            if(len(indices) == 1):
                feature_accuracy_dict[feature_names[indices[0]]] = np.mean(cross_val_score(clf, 
                                        feature_dict[feature_names[indices[0]]], target_label_list,cv=mycv,scoring=score_func))
            elif(len(indices)>1):
                comb_feature_names = ""
                comb_feature_names += feature_names[indices[0]]
                comb_feature = feature_dict[feature_names[indices[0]]]
                
                for i in indices[1:]:
                    cur_feature = feature_dict[feature_names[i]]
                    ext_feature = np.hstack((comb_feature,cur_feature))
                    comb_feature = ext_feature
                    comb_feature_names += " + "+feature_names[i]
                
                feature_accuracy_dict[comb_feature_names] = np.mean(cross_val_score(clf,comb_feature, 
                                                                                target_label_list,cv=mycv,scoring=score_func))
     
    sorted_feature_accu = sorted(feature_accuracy_dict.items(),key=operator.itemgetter(1),reverse=True)
    for item in sorted_feature_accu[:10]:    
        print("%s\t%.3f" % (item[0],item[1]))


def eva_classifier(best_feature,sup_healthlabel_list,mycv,score_func,classifier_list):
    for clf in classifier_list:
        print("%.3f\t%s" % (np.mean(cross_val_score(clf,best_feature,sup_healthlabel_list,cv=mycv,scoring=score_func)),str(clf)))


def healthbrand_sup_predict_matrix(tweet_list,entity,entity_nameID,eco_terms,GN_model,health_keywords,sup_bow_vectorizer,sup_cont_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    
    "BOW_tw_matrix"
    my_bow_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = sup_bow_vectorizer.vocabulary_)
    BOW_tw_matrix = my_bow_vectorizer.fit_transform(tweet_list).toarray()
    
    "BOW_cont_matrix"
    BOW_addCont_tweet = mark_context(tweet_list,eco_terms)
    my_cont_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = sup_cont_vectorizer.vocabulary_)
    BOW_cont_matrix = my_cont_vectorizer.fit_transform(BOW_addCont_tweet).toarray()
    
    "W2V_tw_vector"
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)
    
    "W2V_topnwd_vectors"
    tweet_rankedwd_list = rank_match_words(tweet_list, GN_model, health_keywords)
    W2V_topnwd_vectors = get_topn_vectors(tweet_rankedwd_list,GN_model,n=3)
    
    Best_feature_list =[BOW_tw_matrix, BOW_cont_matrix,W2V_tw_vector,W2V_topnwd_vectors]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix


def healthbrand_predict_label_0_1(sup_classifier,entity_nameid,eco_terms,GN_model,health_keywords,sup_bow_vectorizer,sup_cont_vectorizer,tweet_file,result_file):
    print("Note: this code do prediction for each brand, it takes about 8~10 hours to run for each dataset")
    '''
    Parameters:
        sup_classifier: support classifier
        tweet_file: this file contains all entities' tweets
        result_file: write predicted label (0: non-support, 1: support) to this file
    '''
    with open(tweet_file,'r') as tw_f, open(result_file,"w") as pred_f:
        pred_f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_0","proba_1","predict_label","tweet_text"))
        tw_f.readline()
        ct = 0
        nline = 0
        process_brand = ""
        process_brand_tweet_list = []
        process_tw_score_list = []
        for line in tw_f:
            nline += 1
            if(nline % 1000 == 0):
                print("Processed %d lines." % nline)
            cols = line.strip().split("\t")
            this_brand = cols[0]
            if(this_brand != process_brand):
                ct += 1
                if(len(process_brand_tweet_list)>0):
                    tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
                    predict_feature_matrix = healthbrand_sup_predict_matrix(tweet_list,[process_brand],entity_nameid,eco_terms,GN_model,
                                                                            health_keywords,sup_bow_vectorizer,sup_cont_vectorizer)
                    labels = sup_classifier.predict(predict_feature_matrix)
                    probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
                    for i in range(len(labels)):
                        pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],
                                                            probas[i,1],int(labels[i]),process_brand_tweet_list[i]))
                
                print("processing %d brand: %s" % (ct,this_brand))
                process_brand_tweet_list = []
                process_tw_score_list = []
                process_brand = this_brand
            process_brand_tweet_list.append(cols[3])
            process_tw_score_list.append(cols[2])
        
        #probas,labels = pred_new_data(this_brand_tweet_list,org_vectorizer,org_X7_new, org_label_list)
        
        tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
        predict_feature_matrix = healthbrand_sup_predict_matrix(tweet_list,[process_brand],entity_nameid,eco_terms,GN_model,
                                                                health_keywords,sup_bow_vectorizer,sup_cont_vectorizer)
        labels = sup_classifier.predict(predict_feature_matrix)
        probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
            
        for i in range(len(labels)):
            pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],probas[i,1],
                                                            int(labels[i]),process_brand_tweet_list[i]))
    pred_f.close()
    tw_f.close()    

    print("Test finished!")


def healthbrand_comt_predict_matrix(tweet_list,entity,entity_nameID,eco_terms,GN_model,health_keywords,comt_mention1_vectorizer,comt_mentionAllpron_vectorizer,comt_topn_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    

    "BOW_mentionOnce_matrix"
    BOW_self_tweet_once = selfmention([entity],entity_nameID,tweet_list, count="once")
    my_BOW_mentionOnce_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_mention1_vectorizer.vocabulary_)
    BOW_mentionOnce_matrix = my_BOW_mentionOnce_vectorizer.fit_transform(BOW_self_tweet_once).toarray()
    
    "BOW_mentionAllPron_matrix"
    BOW_addPron_tweet = mark_pronouns(tweet_list, binary = False)
    BOW_selfall_pron_tweet = selfmention([entity],entity_nameID,BOW_addPron_tweet, count="all")
    my_BOW_mentionAllPron_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_mentionAllpron_vectorizer.vocabulary_)
    BOW_mentionAllPron_matrix = my_BOW_mentionAllPron_vectorizer.fit_transform(BOW_selfall_pron_tweet).toarray()
    
    
    "W2V_topnwd_matrix"
    tweet_rankedwd_list = rank_match_words(tweet_list, GN_model, health_keywords)
    W2V_topnwd, W2V_topnwd_scores = get_topn_words(tweet_rankedwd_list,n=3)
    my_W2V_topnwd_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_topn_vectorizer.vocabulary_)
    W2V_topnwd_matrix = my_W2V_topnwd_vectorizer.fit_transform(W2V_topnwd).toarray()
    
    "W2V_topnwd_vectors"
    W2V_topnwd_vectors = get_topn_vectors(tweet_rankedwd_list,GN_model,n=3)
    
    "W2V_topicwd_vectors"
    W2V_topicwd_list, tweet_topicwd_tp_list = get_topic_words(tweet_list,GN_model,health_keywords,threshold = 0.30)
    W2V_topicwd_vectors = get_topicwd_vec(GN_model,tweet_topicwd_tp_list)
    
    Best_feature_list =[BOW_mentionOnce_matrix, BOW_mentionAllPron_matrix,W2V_topnwd_matrix,W2V_topnwd_vectors,W2V_topicwd_vectors]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix


def healthbrand_predict_label_2_3(comt_clf,GN_model,entity_nameID,health_keywords,eco_terms,comt_mention1_vectorizer,comt_mentionAllpron_vectorizer,comt_topn_vectorizer,tweet_file,resfile):
    n=0
    with open(tweet_file,"r") as tw_f, open(resfile,"w") as res:
        tw_f.readline()
        res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_2","proba_3","predict_label","tweet_text"))
        for line in tw_f:
            cols = line.strip().split("\t")
            if(str(cols[4])=="0"):
                res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (cols[0],cols[1],cols[2],cols[3],cols[4],cols[5]))
            elif(str(cols[4]=="1")):
                n += 1
                if(n % 5000 == 0):
                    print("processed %d cases." % n)
                tweet = cols[5]
                
                pred_feature_matrix = healthbrand_comt_predict_matrix([tweet],str(cols[0]),entity_nameID,eco_terms,GN_model,
                                                                      health_keywords,comt_mention1_vectorizer,
                                                                      comt_mentionAllpron_vectorizer,comt_topn_vectorizer)
                
                labels = comt_clf.predict(pred_feature_matrix)
                probas = comt_clf.predict_proba(pred_feature_matrix)
                    
                for i in range(len(labels)):
                    if(labels[i]==0):
                        this_label = 2
                    elif(labels[i]==1):
                        this_label = 3
                    res.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (cols[0],cols[1],probas[i,0],probas[i,1],
                                                            int(this_label),cols[5]))
        res.close()
        tw_f.close()
    print("Test finished")   


def get_aggregate_info(predfile,sim_limit,prob_limit):
    """
    Parameters: 
        predfile: File that contains predicted labels
        sim_limit: set cause-commitment score > 0.3 as relevant candidate tweets
        prob_limit: set probability limit > 0.7 as confident prediction
    Returns:
        entity_pred_info: dict[entity name] = [number of tweets predicted as label 0, number of tweets predicted as label 1,
                                               number of tweets predicted as label 2, number of tweets predicted as label 3,
                                               probability summation of tweets predicted as label 0,
                                               probability summation of tweets predicted as label 2,
                                               probability summation of tweets predicted as label 3] 
        entity_pred2_tw: dict[entity name] = a list of this entity's tweets that are predicted as label 2
        entity_pred3_tw: dict[entity name] = a list of this entity's tweets that are predicted as label 3
    """
    entity_pred_info = {}
    entity_pred2_tw = {}
    entity_pred3_tw = {}
    with open(predfile,"r") as predf:
        predf.readline()
        flag = False
        label0_prob = []
        label2_prob = []
        label3_prob = []
        label0_ct = 0
        label2_ct = 0
        label3_ct = 0
        last_entity = ""
        tw2_list = []
        tw3_list = []
                
        for line in predf:
            cols = line.strip().split("\t")
            if(float(cols[1]) > sim_limit and (float(cols[2])>prob_limit or float(cols[3])>prob_limit)):
                if cols[0] not in entity_pred_info:
                    entity_pred_info[cols[0]] = []
                    entity_pred2_tw[cols[0]] = []
                    entity_pred3_tw[cols[0]] = []
                    if(flag == True):
                        entity_pred_info[last_entity] = [label0_ct,label2_ct,label3_ct,np.sum(label0_prob),np.sum(label2_prob),np.sum(label3_prob)]
                        entity_pred2_tw[last_entity] = tw2_list
                        entity_pred3_tw[last_entity] = tw3_list
                        tw2_list = []
                        tw3_list = []
                        last_entity = cols[0]
                        label0_prob = []
                        label2_prob = []
                        label3_prob = []
                        label0_ct = 0
                        label2_ct = 0
                        label3_ct = 0
                    else:
                        last_entity = cols[0]
                        flag = True
                if(cols[4] == '1'):
                    label0_ct += 1
                    label0_prob.append(float(cols[2]))
                elif(cols[4] == '2'):
                    label2_ct += 1
                    label2_prob.append(float(cols[2]))
                    tw2_list.append(cols[5])
                elif(cols[4] == '3'):
                    label3_ct += 1
                    label3_prob.append(float(cols[3]))
                    tw3_list.append(cols[5])
        
        entity_pred_info[last_entity] = [label0_ct,label2_ct,label3_ct,np.sum(label0_prob),np.sum(label2_prob),np.sum(label3_prob)] 
        entity_pred2_tw[last_entity] = tw2_list
        entity_pred3_tw[last_entity] = tw3_list
    predf.close()
    print("Get data for %d entities" % (len(entity_pred_info)))
    
    return entity_pred_info, entity_pred2_tw, entity_pred3_tw



def filt_entity(entity_predicts,entity_score_dict,ntw_threshold):
    '''
    Parameters:
        entity_predicts: dict of entities prediction information
        entity_score_dict: dict[entity] = third-party rating
        ntw_threshold: threshold for the number of tweets in each entity, ignore entities that have less than threshold tweets
    Return:
        remain_entity_info: dict[remained entity] = prediction information
        remove_entity: a list of removed entities
    '''
    remain_entity_info = {}
    remove_entity = []
    for entity in set(entity_predicts.keys()).intersection(set(entity_score_dict.keys())) :
        ct = int(entity_predicts[entity][0])+int(entity_predicts[entity][1])+int(entity_predicts[entity][2])
        if(entity_score_dict[entity]):
            if(ct >= ntw_threshold) and (float(entity_score_dict[entity])>0.0):
                remain_entity_info[entity] = entity_predicts[entity]
            else:
                remove_entity.append(entity)
    print("%d entities remain after remove entities (number of tweets<%d)" % (len(remain_entity_info),ntw_threshold))
    return remain_entity_info, remove_entity



def aggregation(entity_pred,topn):
    """
    Parameters:
        entity_pred: dict[entity] = prediction information
        topn: number of entities to check
    Return:
        entity_n3: dict[entity] = number of tweets predcited as label 3
        entity_frac3: dict[entity] = fraction of tweets predicted as label 3
        entity_prob3: dict[entity] = average prediction probability of tweets predicted as label 3
        words_topn_entities: intersection of entities in topn of entity_n3, entity_frac3, entity_prob3
    """
    entity_n3 = {}
    entity_frac3 = {}
    entity_prob3 = {}
    for entity in entity_pred:
        entity_n3[entity] = int(entity_pred[entity][2])
        if((int(entity_pred[entity][0])+int(entity_pred[entity][1])+int(entity_pred[entity][2])) > 0):
            entity_frac3[entity] = float(entity_pred[entity][2]/(int(entity_pred[entity][0])+int(entity_pred[entity][1])+int(entity_pred[entity][2])))
        else:
            entity_frac3[entity] = 0.0
        if(int(entity_pred[entity][2])>0):
            entity_prob3[entity] = float(float(entity_pred[entity][5])/int(entity_pred[entity][2]))
        else:
            entity_prob3[entity] = 0.0
    sorted_entity_n3 = sorted(entity_n3.items(), key=operator.itemgetter(1),reverse=True)
    sorted_entity_frac3 = sorted(entity_frac3.items(), key=operator.itemgetter(1),reverse=True)
    sorted_entity_prob3 = sorted(entity_prob3.items(), key=operator.itemgetter(1),reverse=True)
    
    for i in range(topn):
        topn_n3 = [item_n3[0] for item_n3 in sorted_entity_n3[:topn]]
        topn_frac3 = [item_frac3[0] for item_frac3 in sorted_entity_frac3[:topn]]
        topn_prob3 = [item_prob3[0] for item_prob3 in sorted_entity_prob3[:topn]]
    return entity_n3,entity_frac3,entity_prob3,set(set(topn_n3).intersection(set(topn_frac3))).intersection(set(topn_prob3))


def inauthentic(words_topn_entities,entity_score_dict,n):
    """
    Parameters:
        words_topn_entities: a set of entities that are selected as high commitment in words
        entity_score_dict: dict[entity] = third-party ratings
        n: number of inauthentic entities to show
    Return:
        inauthentic_brands: a list of brands that have high word commitment but low action commitment
    """
    action_score = {}
    for entity in words_topn_entities:
        action_score[entity] = entity_score_dict[entity]
    sorted_action_score = sorted(action_score.items(),key=operator.itemgetter(1),reverse=False) 
    return [entity[0] for entity in sorted_action_score[:n]]



def ecobrand_sup_predict_matrix(tweet_list,entity,entity_nameID,eco_terms,GN_model,eco_keywords,sup_cont_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    
    
    "BOW_cont_matrix"
    BOW_addCont_tweet = mark_context(tweet_list,eco_terms)
    my_cont_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = sup_cont_vectorizer.vocabulary_)
    BOW_cont_matrix = my_cont_vectorizer.fit_transform(BOW_addCont_tweet).toarray()
    
    "W2V_tw_vector"
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)
    
    "W2V_topnwd_vectors"
    tweet_rankedwd_list = rank_match_words(tweet_list, GN_model, eco_keywords)
    W2V_topnwd_vectors = get_topn_vectors(tweet_rankedwd_list,GN_model,n=3)
    
    Best_feature_list =[BOW_cont_matrix, W2V_tw_vector,W2V_topnwd_vectors]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix



def ecobrand_predict_label_0_1(sup_classifier,entity_nameid,eco_terms,GN_model,eco_keywords,sup_cont_vectorizer,tweet_file,result_file):
    print("Note: this code do prediction for each brand, it takes several hours to run.")
    '''
    Parameters:
        sup_classifier: support classifier
        tweet_file: this file contains all entities' tweets
        result_file: write predicted label (0: non-support, 1: support) to this file
    '''
    with open(tweet_file,'r') as tw_f, open(result_file,"w") as pred_f:
        pred_f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_0","proba_1","predict_label","tweet_text"))
        tw_f.readline()
        ct = 0
        nline = 0
        process_brand = ""
        process_brand_tweet_list = []
        process_tw_score_list = []
        for line in tw_f:
            nline += 1
            if(nline % 1000 == 0):
                print("Processed %d lines." % nline)
            cols = line.strip().split("\t")
            this_brand = cols[0]
            if(this_brand != process_brand):
                ct += 1
                if(len(process_brand_tweet_list)>0):
                    tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
                    predict_feature_matrix = ecobrand_sup_predict_matrix(tweet_list,[process_brand],entity_nameid,eco_terms,GN_model,
                                                                            eco_keywords,sup_cont_vectorizer)
                    labels = sup_classifier.predict(predict_feature_matrix)
                    probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
                    for i in range(len(labels)):
                        pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],
                                                            probas[i,1],int(labels[i]),process_brand_tweet_list[i]))
                
                print("processing %d brand: %s" % (ct,this_brand))
                process_brand_tweet_list = []
                process_tw_score_list = []
                process_brand = this_brand
            process_brand_tweet_list.append(cols[3])
            process_tw_score_list.append(cols[2])
        
        #probas,labels = pred_new_data(this_brand_tweet_list,org_vectorizer,org_X7_new, org_label_list)
        
        tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
        predict_feature_matrix = ecobrand_sup_predict_matrix(tweet_list,[process_brand],entity_nameid,eco_terms,GN_model,
                                                                eco_keywords,sup_cont_vectorizer)
        labels = sup_classifier.predict(predict_feature_matrix)
        probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
            
        for i in range(len(labels)):
            pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],probas[i,1],
                                                            int(labels[i]),process_brand_tweet_list[i]))
    pred_f.close()
    tw_f.close()    

    print("Test finished!")


def ecobrand_comt_predict_matrix(tweet_list,entity,entity_nameID,eco_terms,GN_model,eco_keywords,comt_bow_vectorizer,comt_mention1_vectorizer,comt_mentionAllpron_vectorizer,comt_rmtopic_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    
    "BOW_tw_matrix"
    my_bow_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_bow_vectorizer.vocabulary_)
    BOW_tw_matrix = my_bow_vectorizer.fit_transform(tweet_list).toarray()
    
    "BOW_mentionOnce_matrix"
    BOW_self_tweet_once = selfmention([entity],entity_nameID,tweet_list, count="once")
    my_mention1_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_mention1_vectorizer.vocabulary_)
    BOW_mentionOnce_matrix = my_mention1_vectorizer.fit_transform(BOW_self_tweet_once).toarray()
    
    
    "BOW_mentionAllPron_matrix"
    BOW_addPron_tweet = mark_pronouns(tweet_list, binary = False)
    BOW_selfall_pron_tweet = selfmention([entity],entity_nameID,BOW_addPron_tweet, count="all")
    my_mentionAllPron_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_mentionAllpron_vectorizer.vocabulary_)
    BOW_mentionAllPron_matrix = my_mentionAllPron_vectorizer.fit_transform(BOW_selfall_pron_tweet).toarray()
    

    "BOW_rmTopic_matrix"
    BOW_rmTopic_tweet = remove_keywords(tweet_list, eco_terms)
    my_rmtopic_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_rmtopic_vectorizer.vocabulary_)
    BOW_rmTopic_matrix = my_rmtopic_vectorizer.fit_transform(BOW_rmTopic_tweet).toarray()
    
    "W2V_tw_vector"
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)
    
    
    
    Best_feature_list =[BOW_tw_matrix, BOW_mentionOnce_matrix,BOW_mentionAllPron_matrix,BOW_rmTopic_matrix,W2V_tw_vector]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix


def ecobrand_predict_label_2_3(comt_clf,GN_model,entity_nameID,eco_terms,eco_keywords,comt_bow_vectorizer,comt_mention1_vectorizer,comt_mentionAllpron_vectorizer,comt_rmtopic_vectorizer,tweet_file,resfile):
    n=0
    with open(tweet_file,"r") as tw_f, open(resfile,"w") as res:
        tw_f.readline()
        res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_2","proba_3","predict_label","tweet_text"))
        for line in tw_f:
            cols = line.strip().split("\t")
            if(str(cols[4])=="0"):
                res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (cols[0],cols[1],cols[2],cols[3],cols[4],cols[5]))
            elif(str(cols[4]=="1")):
                n += 1
                if(n % 5000 == 0):
                    print("processed %d cases." % n)
                tweet = cols[5]
                
                pred_feature_matrix = ecobrand_comt_predict_matrix([tweet],str(cols[0]),entity_nameID,eco_terms,GN_model,
                                                                      eco_keywords,comt_bow_vectorizer,comt_mention1_vectorizer,comt_mentionAllpron_vectorizer,comt_rmtopic_vectorizer)
                
                labels = comt_clf.predict(pred_feature_matrix)
                probas = comt_clf.predict_proba(pred_feature_matrix)
                    
                for i in range(len(labels)):
                    if(labels[i]==0):
                        this_label = 2
                    elif(labels[i]==1):
                        this_label = 3
                    res.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (cols[0],cols[1],probas[i,0],probas[i,1],
                                                            int(this_label),cols[5]))
        res.close()
        tw_f.close()
    print("Test finished")


def ecoMOC_sup_predict_matrix(tweet_list,entity_list,entity_nameID,eco_terms,GN_model,eco_keywords,sup_mentionAllpron_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    
    "BOW_mentionAllPron_matrix"
    BOW_addPron_tweet = mark_pronouns(tweet_list, binary = False)
    BOW_selfall_pron_tweet = selfmention(entity_list,entity_nameID,BOW_addPron_tweet, count="all")
    my_mentionAllPron_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = sup_mentionAllpron_vectorizer.vocabulary_)
    BOW_mentionAllPron_matrix = my_mentionAllPron_vectorizer.fit_transform(BOW_selfall_pron_tweet).toarray()
    
    
    "W2V_tw_vector"
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)
    
    "W2V_topnwd_vectors"
    tweet_rankedwd_list = rank_match_words(tweet_list, GN_model, eco_keywords)
    W2V_topnwd_vectors = get_topn_vectors(tweet_rankedwd_list,GN_model,n=3)
    
    "W2V_topicwd_vectors"
    W2V_topicwd_list, tweet_topicwd_tp_list = get_topic_words(tweet_list,GN_model,eco_keywords,threshold = 0.30)
    W2V_topicwd_vectors = get_topicwd_vec(GN_model,tweet_topicwd_tp_list)
    
    Best_feature_list =[BOW_mentionAllPron_matrix, W2V_tw_vector,W2V_topnwd_vectors,W2V_topicwd_vectors]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix



def ecoMOC_predict_label_0_1(sup_classifier,entity_nameid,eco_terms,GN_model,eco_keywords,sup_mentionAllpron_vectorizer,tweet_file,result_file):
    print("Note: this code do prediction for each congress member, it takes several hours to run")
    '''
    Parameters:
        sup_classifier: support classifier
        tweet_file: this file contains all entities' tweets
        result_file: write predicted label (0: non-support, 1: support) to this file
    '''
    with open(tweet_file,'r') as tw_f, open(result_file,"w") as pred_f:
        pred_f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_0","proba_1","predict_label","tweet_text"))
        tw_f.readline()
        ct = 0
        nline = 0
        process_brand = ""
        process_brand_tweet_list = []
        process_tw_score_list = []
        for line in tw_f:
            nline += 1
            if(nline % 1000 == 0):
                print("Processed %d lines." % nline)
            cols = line.strip().split("\t")
            this_brand = cols[0]
            if(this_brand != process_brand):
                ct += 1
                if(len(process_brand_tweet_list)>0):
                    tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
                    predict_feature_matrix = ecoMOC_sup_predict_matrix(tweet_list,[process_brand]*len(tweet_list),entity_nameid,eco_terms,GN_model,
                                                                            eco_keywords,sup_mentionAllpron_vectorizer)
                    
                    labels = sup_classifier.predict(predict_feature_matrix)
                    probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
                    for i in range(len(labels)):
                        pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],
                                                            probas[i,1],int(labels[i]),process_brand_tweet_list[i]))
                
                print("processing %d brand: %s" % (ct,this_brand))
                process_brand_tweet_list = []
                process_tw_score_list = []
                process_brand = this_brand
            process_brand_tweet_list.append(cols[3])
            process_tw_score_list.append(cols[2])
        
        #probas,labels = pred_new_data(this_brand_tweet_list,org_vectorizer,org_X7_new, org_label_list)
        
        tweet_list = [' '.join(tw_tokenize_with_features(tw)) for tw in process_brand_tweet_list]
        predict_feature_matrix = ecoMOC_sup_predict_matrix(tweet_list,[process_brand]*len(tweet_list),entity_nameid,eco_terms,GN_model,
                                                                eco_keywords,sup_mentionAllpron_vectorizer)
        labels = sup_classifier.predict(predict_feature_matrix)
        probas = sup_classifier.predict_proba(predict_feature_matrix)
                    
            
        for i in range(len(labels)):
            pred_f.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (process_brand,process_tw_score_list[i],probas[i,0],probas[i,1],
                                                            int(labels[i]),process_brand_tweet_list[i]))
    pred_f.close()
    tw_f.close()    

    print("Test finished!")

def ecoMOC_comt_predict_matrix(tweet_list,entity_list,entity_nameID,eco_terms,GN_model,eco_keywords,comt_BOW_vectorizer,comt_mentionAllpron_vectorizer):
    """
    Parameters:
        entity: screenname of a brand or a congress member
        entity_nameID: dict[entity's screen name] = entity's unique real name
        tweet_list: a list of tweets of this entity
    Return:
        pred_feature_matrix: The constructed feature matrix used for prediction.
                             Feature matrix is constructed based on the best feature set for support classification of the dataset
    """
    
    
    "BOW_tw_matrix"
    my_bow_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_BOW_vectorizer.vocabulary_)
    BOW_tw_matrix = my_bow_vectorizer.fit_transform(tweet_list).toarray()
    
    
    
    "BOW_mentionAllPron_matrix"
    BOW_addPron_tweet = mark_pronouns(tweet_list, binary = False)
    BOW_selfall_pron_tweet = selfmention(entity_list,entity_nameID,BOW_addPron_tweet, count="all")
    my_mentionAllPron_vectorizer = CountVectorizer(stop_words = "english",tokenizer = tw_tokenize_with_features,
                                    vocabulary = comt_mentionAllpron_vectorizer.vocabulary_)
    BOW_mentionAllPron_matrix = my_mentionAllPron_vectorizer.fit_transform(BOW_selfall_pron_tweet).toarray()
    
    
    "W2V_tw_vector"
    W2V_tw_vector = construct_tw_vector(tweet_list,GN_model)
    
    
    
    Best_feature_list =[BOW_tw_matrix, BOW_mentionAllPron_matrix, W2V_tw_vector]
    pred_feature_matrix = Best_feature_list[0]
    for feature in Best_feature_list[1:]:
        pred_feature_matrix = np.hstack((pred_feature_matrix,feature))

    return  pred_feature_matrix


def ecoMOC_predict_label_2_3(comt_clf,GN_model,entity_nameID,eco_terms,eco_keywords,comt_BOW_vectorizer,comt_mentionAllpron_vectorizer,tweet_file,resfile):
    n=0
    with open(tweet_file,"r") as tw_f, open(resfile,"w") as res:
        tw_f.readline()
        res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("brand","twscore","proba_2","proba_3","predict_label","tweet_text"))
        for line in tw_f:
            cols = line.strip().split("\t")
            if(str(cols[4])=="0"):
                res.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (cols[0],cols[1],cols[2],cols[3],cols[4],cols[5]))
            elif(str(cols[4]=="1")):
                n += 1
                if(n % 5000 == 0):
                    print("processed %d cases." % n)
                tweet = cols[5]
                
                pred_feature_matrix = ecoMOC_comt_predict_matrix([tweet],[str(cols[0])],entity_nameID,eco_terms,GN_model,
                                                                      eco_keywords,comt_BOW_vectorizer,comt_mentionAllpron_vectorizer)
                
                labels = comt_clf.predict(pred_feature_matrix)
                probas = comt_clf.predict_proba(pred_feature_matrix)
                    
                for i in range(len(labels)):
                    if(labels[i]==0):
                        this_label = 2
                    elif(labels[i]==1):
                        this_label = 3
                    res.write("%s\t%s\t%.3f\t%.3f\t%d\t%s\n" % (cols[0],cols[1],probas[i,0],probas[i,1],
                                                            int(this_label),cols[5]))
        res.close()
        tw_f.close()
    print("Test finished")


def show_moc_data(data):    
    df = pd.DataFrame(json.loads(s) for s in open(data))
    def pct2float(x):
        return float(x.strip('%')) / 100
    df['mocLifetimeRating'] = df['mocLifetimeRating'].apply(pct2float)
    df['num_tweets'] = [len(x) for x in df.tweets]
    return df

