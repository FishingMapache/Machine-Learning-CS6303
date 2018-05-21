import json
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re, string, sys

def find_new_cen(tweets, clusters, id_with_clusters, jaccard_matrix, k):
    new_clusters, new_ids = {}, {}
    for k in range(k):
        new_clusters[k] = set()

    for t in tweets:
        min_dist = float("inf")
        min_cluster = id_with_clusters[t]

        for k in clusters:
            dist, total = 0, 0
            for t2 in clusters[k]:
                dist += jaccard_matrix[t][t2]
                total += 1
            if total > 0:
                average_dist = float(dist / float(total))
                if min_dist > average_dist:
                    min_dist = average_dist
                    min_cluster = k
                new_ids[t] = min_cluster
        new_clusters[min_cluster].add(t)

    return new_clusters, new_ids

def initialize_clusters(tweets, seeds, k):
    clusters, ids = {}, {}
    for ID in tweets: ids[ID] = -1

    for k in range(k):
        clusters[k] = set([seeds[k]])
        ids[seeds[k]] = k

    return clusters, ids

def bag_of_words(tweet_text):
    stop_words = stopwords.words("english")
    tweets_lower_case = tweet_text.lower()
    line = tweets_lower_case.split(" ")
    sentence = []
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    for word in line:
        word = word.strip()
        if not re.match(r'^https?:\/\/.*[\r\n]*', word) and word != '' and not re.match('\s', word) and word != 'rt' and not re.match('^@.*', word) and word not in stop_words:
            clean_word = regex.sub("", word)
            sentence.append(clean_word)

    return sentence

def jaccard_dist_table(tweets):
    jaccard_table = {}
    for t in tweets:
        jaccard_table[t] = {}
        b1 = set(bag_of_words(tweets[t]["text"]))

        for t2 in tweets:
            if t2 not in jaccard_table:
                jaccard_table[t2] = {}
            b2 = set(bag_of_words(tweets[t2]["text"]))
            jaccard_dist = 1 - float(len(b1.intersection(b2))) / float(len(b1.union(b2)))
            jaccard_table[t][t2], jaccard_table[t2][t] = jaccard_dist, jaccard_dist

    return jaccard_table

def kMeans(seeds, tweets, num_centroids):
    jaccard_table = jaccard_dist_table(tweets)
    clusters, ids = initialize_clusters(tweets, seeds, num_centroids)

    new_clusters, new_ids = find_new_cen(tweets, clusters, ids, jaccard_table, num_centroids)
    clusters = new_clusters
    ids = new_ids

    i = 1
    while i < max_iter:
        new_clusters, new_ids = find_new_cen(tweets, clusters, ids, jaccard_table, num_centroids)
        i = i + 1
        if ids != new_ids:
            clusters = new_clusters
            ids = new_ids
        else:
            return clusters

    return clusters

if __name__ == "__main__":
    # Run in terminal with following command:
    # python tweets-k-means.py "25" "Directory of InitialSeeds.txt" "Directory of Tweets.json" "Direcotry of output.txt"
    # e.g. python tweets-k-means.py "25" "/Users/DianaZi/Desktop/InitialSeeds.txt" "/Users/DianaZi/Desktop/Tweets.json" "/Users/DianaZi/Desktop/output.txt"

    max_iter = 5
    num_centroids, seed_path, tweets_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    tweets = {}
    with open(tweets_path, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets[tweet['id']] = tweet
    seeds =[]
    with open(seed_path) as output:
        for line in output:
            seeds.append(int(line.rstrip(',\n')))
    output.close()

    clusters = kMeans(seeds, tweets, int(num_centroids))

    output_file = open(output_path, "w")
    for k in clusters:
        write_line = str(k) + '\t'
        for tweetID in clusters[k]:
            write_line += str(tweetID) + ", "
        output_file.write(write_line + "\n")
    output_file.close()