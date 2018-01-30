These files are results for entities' cause commitment. 
Entities refer to brands from GoodGuide and Member of Congress (MOC for short);
Causes refer to eco and health.
The paper experiment with brands' commitment to causes eco and health, and MOC's commitment to cause eco.

(1)Files that contain brands with corresponding cause scores:
GoodGuide_ecobrands.csv, data field "TGS" represents the brand's eco score;
GoodGuide_healthbrands.csv, data field "health score" represents the brand's health score;
eco_terms.txt: contain 113 terms that relate with cause eco.

(2)Files that contain entities' tweets:
tweets.pruned.json.gz, tweets collected for brands.
congress_pruned.json, tweets collected for MOC.

(3)Files that contain human labeled tweets for each entity:
brand_eco_labeled_tweets.csv
brand_health_labeled_tweets.csv
congress_eco_labeled_tweets.csv

In each file, different data fields mean:
Column1: "eco-brand/health-brand/eco-moc", brand names or MOC names;
Column2: "relevance-score", relevance between tweet and cause, defined as cosine similarity between tweet vector and cause vector, ranges from -1~1;
Column3: "label", human label for tweet text, 0=off topic; 1=on topic, but not indicating support; 2=on topic, and low commitment; 3=on topic, and high commitment;
Column4: "tweet", tweet text.

(4)Files that contain classification result for each tweet:
Health_brand_tweet_predict_proba_01_2_3.txt
Eco_brand_tweet_predict_proba_01_2_3.txt
Eco_moc_tweet_predict_proba_01_2_3.txt

Column1: "brand", brand name or MOC name;
Column2: "twscore", cosine similarity between tweet vector and cause vector, ranges from -1~1;
Column3: "proba_0", predicted probability for negative class(for 0,1 classification, 0 is negative and 1 is positive; for 2,3 classification, 2 is negative and 3 is positive); 
Column4: "proba_1", predicted probability for positive class;	
Column5: "predict_label", predicted label for each tweet (mixed 0 and 1 in this file, so there are 3 predicted labels, 1(represents 0/1),2,3);
Column6: "tweet_text".

Note: if predict_label = 1, then the 1 is a sign that represents either label 0 or label 1. If proba_0 > proba_1, then actual_label=0, viceversa.
If predict_label = 2/3, then proba_0 represents the predicted probability for class label 2 and proba_1 represents the predicted probability for class label 3.


(5)Files that contain statistical information for each entity:
Eco_brands_tweet_commitment.txt
Eco_mocs_tweet_commitment.txt
Health_brands_tweet_commitment.txt

Datafields in these 3 files are statistics filtered based on classification results, and these data are used to fit the linear regression model for aggregating to get the final commitment score of each entity.
Column1: "brand", brand name or MOC name;
Column2: "score", brands' eco score / brands' health score / MOC's eco score;
Column3: "n_tweets_no_limit", total number of tweets collected for this entity;
Column4: "n_tweets_similarity_limit", number of tweets that have cosine_similarity(tweet vector, cause vector) > 0.3; 	
Column5: "n_tweets_similarity_proba_limit", number of tweets that have cosine_similarity(tweet vector, cause vector) > 0.3 and predict probability > 0.7;
Column6: "n_c1", number of tweets predicted as class label 1;
Column7: "n_c2", number of tweets predicted as class label 2;
Column8: "n_c3", number of tweets predicted as class label 3;
Column9: "n_c1_thresh", number of tweets predicted as class label 1 with predict probability > 0.7;	
Column10: "n_c2_thresh", number of tweets predicted as class label 2 with predict probability > 0.7;
Column11: "n_c3_thresh", number of tweets predicted as class label 3 with predict probability > 0.7.

