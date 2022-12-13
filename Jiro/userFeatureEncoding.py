import json
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import vocabulary

CONTRACTION_MAP = {"ain't": "is not","aren't": "are not","can't": "cannot","can't've": "cannot have",
    "'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
    "didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
    "hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
    "he'll've": "he he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will",
    "how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have",
    "I'm": "I am","I've": "I have","i'd": "i would","i'd've": "i would have","i'll": "i will","i'll've": "i will have",
    "i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will",
    "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not",
    "might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have",
    "mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have",
    "o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not",
    "sha'n't": "shall not","shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
    "she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so as",
    "that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there would",
    "there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have",
    "they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we would","we'd've": "we would have","we'll": "we will",
    "we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not","what'll": "what will",
    "what'll've": "what will have","what're": "what are","what's": "what is","what've": "what have",
    "when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have",
    "who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
    "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
    "would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
    "y'all've": "you all have","you'd": "you would","you'd've": "you would have","you'll": "you will",
    "you'll've": "you will have","you're": "you are","you've": "you have"}

# keep specific punctuation and split the sentences
def keepPunctuation(t):
    return re.findall(r"[\w']+|[.!?]", t)


# replace contracted word with uncontracted words
def uncontract(wArray):
    for i in range(len(wArray)):
        # if there is a contraction, replace the word and insert the rest
        if wArray[i] in CONTRACTION_MAP.keys():
            newWords = CONTRACTION_MAP[wArray[i]].split()
            wArray[i] = newWords[0]
            for j in range(1, len(newWords)):
                wArray.insert(i+j, newWords[j])
    return wArray


# remove common stopwords from the word array
def removeStopwords(wArray):
    return [words for words in wArray if words.lower() not in stopwords.words('english')]


# lemmatizes words into simpler form
def lemmatizer(wArray):
    return [WordNetLemmatizer().lemmatize(words) for words in wArray]

# attempts Bernet's vocabulary encoder as detailed https://thedatafrog.com/en/articles/text-preprocessing-machine-learning-yelp/
def orderedEncoder(wArray, vocab, index):
    for word in wArray:
        if word not in index:
            vocab.append(word)
            index[word] = len(vocab) - 1
    
    return vocab, index



# pre-processes all reviews into review-based user feature vectors
def processReviews(fLoc, eLim, n, wLim, save, saveLoc):

    # process json into raw pandas data
    rFile = open(fLoc, "r", encoding='utf-8') 
    allData = list()
    stop = int(n)
    for i, line in enumerate(rFile):
        if i==stop:
            break    
        # convert the json on this line to a dict
        data = json.loads(line)
        # extract what we want
        id = data['user_id']
        business = data['business_id']
        stars = data['stars']
        text = data['text']
        useful = data['useful']
        funny = data['funny']
        cool = data['cool']

        # add to the data collected so far
        allData.append([id, stars, useful, funny, cool, text])
    # create the DataFrame
    rFile.close()
    df = pd.DataFrame(allData, columns=['user_id','business_id','stars', 'useful', 'text'])
    
    # create the feature vocabulary and indicies
    vocab = []
    index = {}
    count = 0
    print("processing texts...")
    for text in df.text:
        wordList = keepPunctuation(text)
        wordList = uncontract(wordList)
        wordList = removeStopwords(wordList)
        wordList = lemmatizer(wordList)
        wordList = wordList[:wLim]
        # df.at[count, 'text'] = wordList
        vocab, index = orderedEncoder(wordList, vocab, index)
        count += 1
        if count % 1000 == 0:
            print(count)
    vocab = np.array(vocab)
    np.save('vocab.npy', vocab)
    # create json object from dictionary
    kson = json.dumps(index)

    # open file for writing, "w" 
    f = open("indexdict.json","w")

    # write json object to file
    f.write(kson)

    # close file
    f.close()

    # #######################################################
    # ## OPTIONAL SKIP THE VOCAB PHASE BECAUSE ITS LONG START (not completed)
    # #######################################################
    # with open("indexdict.json","r") as file:
    #     print(file)
    #     index = json.load(file)
    # print(index)
    # vocab = np.load('vocab.npy')

    # ######################################################
    # ## OPTIONAL SKIP THE VOCAB PHASE BECAUSE ITS LONG END (not completed)
    # #####################################################
    

    # for each unique user, create a feature vector created by at most 5 reviews
    # ...sorted by most useful to least

    print("encoding user features...")
    features = []
    for user in df.user_id.unique():
        userRows = df.loc[df['user_id'] == user].sort_values(by=['useful'])
        currFeat = [user]
        textCount = 0
        for indie, row in userRows.iterrows():
            currFeat.append(row['stars'])
            currFeat.append(row['useful'])
            currFeat.append(row['funny'])
            currFeat.append(row['cool'])
            for word in row['text']:
                currFeat.append(index[word])
            remainingWords = wLim - len(row['text'])
            for i in range(remainingWords):
                currFeat.append(0)
            textCount += 1
        if textCount < eLim:
            currFeat.extend([0 for k in range(wLim+4)])
            textCount += 1
        features.append(currFeat)

    features = np.array(features)
    np.save('features.npy', features)
    return  


# the option parser for encoding parameters
def parseArgs():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-l", "--location", dest="location",
                        action = "store", type = "string",
                        default = "../yelp_dataset/yelp_academic_dataset_review.json",
                        help="write review data file location")
    parser.add_option("-n", "--number", dest="numberEntries",
                        action = "store", type = "int",
                        default = 10000,
                        help="max number of reviews considered for forming vectors")
    parser.add_option("-w", "--words", dest="wordLimit",
                        action = "store", type = "int",
                        default = 97,
                        help="max number of word counts in each review considered for forming vectors")
    parser.add_option("-e", "--encoding", dest="encodingLimit",
                        action = "store", type = "int",
                        default = 5,
                        help="max number of user reviews considered for encoding in each vector")
    parser.add_option("-s", "--save", dest="save",
                        action = "store_true",
                        help="whether to save the newly encoded feature vectors")
    parser.add_option("-d", "--dest", dest="destination",
                        action = "store", type = "string",
                        default = "../yelp_dataset/encodedFeatures.txt",
                        help="save encoded features at file location if saving is applied")
    # for now the bottom is too difficult so I'm not going to try
    # parser.add_option("-p", "--parallel", dest="parallelize",
    #                     action = "store", type = "boolean",
    #                     default = "False",
    #                     help="whether to run the script parallel on multiple cores ")

    return parser.parse_args()

# main function
if __name__ == '__main__':
    options, args = parseArgs()
    processReviews(options.location, options.encodingLimit, options.numberEntries,
                    options.wordLimit, False if options.save==None else options.save,
                    options.destination)
