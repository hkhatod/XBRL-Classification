# import nltk



# if __name__ =="__main__":
#     f = open("rows.csv", "r")
#     inputfile = f.read()
#     tokens = nltk.tokenize.word_tokenize(inputfile)
#     fd = nltk.FreqDist(tokens)
#     #fd.plot(30,cumulative=False)
#     print(fd.values())


from collections import Counter


if __name__ =="__main__":
    csv_file='./training/pickles/standard and documentation/training_sets/SFP/AssetsCurrent.csv'
    with open(csv_file) as input_file:
        lines = [line.split(",", 2) for line in input_file.readlines()]
    text_list = [" ".join(line) for line in lines]
    dictionary = Counter(text_list)
print(dictionary.most_common())