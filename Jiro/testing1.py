import json
import pandas as pd
import matplotlib.pyplot as plt
import re

hello = re.findall(r"[\w']+|[.!?]", "Yo... What's up my dudes: Hello, I'm a string!!")
print(hello)
fuckeroo = []

def fuck(boo):
    boo.append("yeppeee")
    return

fuck(fuckeroo)
print(fuckeroo)

# # open input file: 
# reviewF = '../yelp_dataset/yelp_academic_dataset_review.json'
# userF = '../yelp_dataset/yelp_academic_dataset_user.json'

# fileName = reviewF

# # ifile = open(fileName, "r", encoding='utf-8') 

# # # read the first 100k entries
# # # set to -1 to process everything
# # num_lines = sum(1 for line in ifile)
# # print(num_lines)
# # ifile.close()
# ifile = open(fileName, "r", encoding='utf-8') 
# allData = list()
# stop = -1
# for i, line in enumerate(ifile):
#     if i%10000==0:
#         print(i)
#     if i==stop:
#         break    
#     # convert the json on this line to a dict
#     data = json.loads(line)
#     # extract what we want
#     id = data['user_id']
#     stars = data['stars']
#     text = len(data['text'].split())
#     useful = data['useful']
#     funny = data['funny']
#     cool = data['cool']

#     # add to the data collected so far
#     allData.append([id, stars, text, useful, funny, cool])
# # create the DataFrame
# ifile.close()
# df = pd.DataFrame(allData, columns=['user_id','stars', 'text', 'useful', 'funny', 'cool'])

# print(df)
# # df.to_hdf('revie20ws.h5','reviews')

# ifile.close()

# rc = df['text'] 
# # sorted(rc.unique())
# print("Max")
# print(rc.max())
# print("Min")
# print(rc.min())
# print("Mean")
# print(rc.mean())
# print("Median")
# print(rc.median())
# print("STD")
# print(rc.std())
# # # [1.0, 2.0, 3.0, 4.0, 5.0]

# # plt.hist(stars, range=(0.5, 5.5), bins=5)
# # plt.show()
# # plt.clf() # clear previous figure
# # plt.hist(df['text'].str.len(), bins=100)
# # plt.show()

# # df.loc[ df['text'].str.len() == df['text'].str.len().max() ]


# # # ignore all these words that don't do anything
# # stopwords = set(['.','i','a','and','the','to', 'was', 'it', 'of', 'for', 'in', 'my', 
# #                  'that', 'so', 'do', 'our', 'the', 'and', ',', 'my', 'in', 'we', 'you', 
# #                  'are', 'is', 'be', 'me'])