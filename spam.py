import matplotlib.pyplot as plt
import csv,pandas,sklearn
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 

df = [line.rstrip() for line in open('/home/rishabh/smsspamcollection/SMSSpamCollection')]
#print len(df)

df = pandas.read_csv('/home/rishabh/smsspamcollection/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,names=["Status", "Message"])

'''
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

#print messages.message.head().apply(split_into_lemmas)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
#print len(bow_transformer.vocabulary_)

messages_bow = bow_transformer.transform(messages['message'])
#print 'sparse matrix shape:', messages_bow.shape
#print 'number of non-zeros:', messages_bow.nnz
#print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)


spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
all_predictions = spam_detector.predict(messages_tfidf)
print 'Naive Bayes accuracy', accuracy_score(messages['label'], all_predictions)

spam_detector = SVC().fit(messages_tfidf, messages['label'])
all_predictions = spam_detector.predict(messages_tfidf)
print 'SVM accuracy', accuracy_score(messages['label'], all_predictions)

print "--------------NB----------------------"
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

pipeline_nb = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    					('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    					('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
						])

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline_nb,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

nb_detector = grid.fit(msg_train, label_train)
print grid.score(msg_test,label_test)
print nb_detector.predict(["Can i talk to u?"])[0]
print nb_detector.predict(["Dear Customer, know about best fit offers"])[0]
print nb_detector.predict(["call our customer."])[0]
print nb_detector.predict(["I am a winner!!"])[0]

print "--------------SVM----------------------"
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector = grid_svm.fit(msg_train, label_train)
print grid_svm.score(msg_test,label_test)
print grid_svm.predict(["Dear Customer, know about best offers"])[0]
print grid_svm.predict(["Hello there!!"])[0]
print grid_svm.predict(["call our customer."])[0]
'''

#print len(df[df.Status=='spam'])

df.loc[df["Status"]=='ham',"Status"]=1
df.loc[df["Status"]=='spam',"Status"]=0

#print df.head()

df_x = df["Message"]
df_y = df["Status"]

cv1 = TfidfVectorizer(min_df=1,stop_words='english')

x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2)

x_traincv = cv1.fit_transform(x_train)
a = x_traincv.toarray()

mnb = MultinomialNB()

y_train = y_train.astype('int')

mnb.fit(x_traincv,y_train)

x_testcv = cv1.transform(x_test)
print type(x_test)

pred = mnb.predict(x_testcv)

actual = np.array(y_test)

count = 0
for i in range(len(pred)):
    if pred[i] == actual[i]:
        count+=1

print count/float(len(pred))