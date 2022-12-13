import json
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

# pre-processes all reviews into review-based user feature vectors
def preprocessBert(fLoc, eLim, n, wLim, save, saveLoc, oneReview=True):

    # process json into raw pandas data

    test = np.load('features.npy')
    print(test)
    return
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
        allData.append([id, business, stars, text, useful])
    # create the DataFrame
    rFile.close()
    df = pd.DataFrame(allData, columns=['user_id','business_id','stars', 'text', 'useful'])
    

    newData = pd.DataFrame()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    features = []
    for index, row in df.iterrows():
        currFeat = [row['user_id']]
        currFeat.append(row['business_id'])
        currFeat.append(row['stars'])
        text = row['text']
        text = "[CLS] " + row['text'] + " [SEP]"
        tokenizedText = tokenizer.tokenize(text, padding=True)

        # truncate from middle because https://arxiv.org/abs/1905.05583
        # says its more effective than either beginnning or end
        if len(tokenizedText) > 512:
            toTruncate = len(tokenizedText) - 512
            for word in range(toTruncate):
                goodbye = len(tokenizedText) // 2
                if tokenizedText[goodbye] != "[SEP]":
                    tokenizedText.pop(goodbye)
                else:
                    tokenizedText.pop(goodbye-1)
        currFeat.append(tokenizedText)
        features.append(currFeat)
        
    features = np.array(features)
    # print(features[:1])
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
                        default = 100000, # default is 10000
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
    preprocessBert(options.location, options.encodingLimit, options.numberEntries,
                    options.wordLimit, False if options.save==None else options.save,
                    options.destination)
