import sys

def distinct_1(path):
    inFile = open(path, mode="r", encoding="utf8")
    char_set = set()
    all_unigram_count = 0
    for line in inFile.readlines():
        line = line.strip().split(" ")
        for word in line:
            char_set.add(word)
        all_unigram_count += len(line)
    distinct_unigram_count = len(char_set)
    print("distinct_unigram: ", distinct_unigram_count)
    print("all_unigram: ", all_unigram_count)
    print("distinct 1: " + str(distinct_unigram_count / all_unigram_count))
    inFile.close()
    return distinct_unigram_count / all_unigram_count

sp="#####"

def distinct_2(path):
    inFile = open(path, mode="r", encoding="utf8")
    bichar_set = set()
    all_bigram_count = 0
    for line in inFile.readlines():
        line = line.strip().split(" ")
        char_len = len(line)
        for idx in range(char_len - 1):
            bichar_set.add(line[idx] + sp + line[idx + 1])
        all_bigram_count += (char_len - 1)

    distinct_bigram_count = len(bichar_set)
    print("distinct_bigram: ", distinct_bigram_count)
    print("all_bigram: ", all_bigram_count)
    print("distinct 2: " + str(distinct_bigram_count / all_bigram_count))
    inFile.close()
    return distinct_bigram_count / all_bigram_count

def distinct_3(path):
    inFile = open(path, mode="r", encoding="utf8")
    bichar_set = set()
    all_bigram_count = 0
    for line in inFile.readlines():
        line = line.strip().split(" ")
        char_len = len(line)
        for idx in range(char_len - 2):
            bichar_set.add(line[idx] + sp + line[idx + 1] + sp + line[idx + 2])
        all_bigram_count += (char_len -2)

    distinct_bigram_count = len(bichar_set)
    print("distinct_trigram: ", distinct_bigram_count)
    print("all_trigram: ", all_bigram_count)
    print("distinct 3: " + str(distinct_bigram_count / all_bigram_count))
    inFile.close()
    return distinct_bigram_count / all_bigram_count

distinct_1(sys.argv[1])
distinct_2(sys.argv[1])
distinct_3(sys.argv[1])