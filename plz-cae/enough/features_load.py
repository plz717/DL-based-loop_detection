import pickle

#load the file containing features_list
with open("test.txt","rb") as fp:
        feature_array_list=pickle.load(fp)
