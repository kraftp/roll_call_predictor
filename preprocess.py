#from __future__ import print_function
import numpy as np
import h5py
import argparse
import sys
import os
import json
import csv
from sklearn.cross_validation import train_test_split, KFold
from time import time
import gzip
import re

args = {}

AYE = 3  # Also Yea
NAY = 2  # Also No
PRES = 1 # Also Not Voting
NA = 0   # Not listed at all (not in relevant chamber/Congress)

def gen_congressperson_bill_dict(data):
    print("Generating congressperson_dict and bill_dict")
    i = 0
    j = 0
    congressperson_dict = {}
    bill_dict = {}
    for root, _, files in os.walk(data):
        if "votes" in root and "data.json" in files:
            json_data = json.loads(open(root + "/data.json").read())
            if "passage" in json_data["category"] and "bill" in json_data:
                congress_num = root.split(os.sep)[1]
                b_id = congress_num + json_data["bill"]["type"] + str(json_data["bill"]["number"])
                if b_id not in bill_dict:
                    bill_dict[b_id] = j
                    j += 1
                for category in json_data["votes"]:
                    for entry in json_data["votes"][category]:
                        try:
                            c_id = entry["id"]
                        except:
                            print "probably the vice president: {0}".format(entry)
                        if c_id not in congressperson_dict:
                            congressperson_dict[c_id] = i
                            i += 1
    return congressperson_dict, bill_dict

def gen_vote_matrix(data, cp_dict, bill_dict):
    print("Generating vote_matrix")
    vote_matrix = np.zeros((len(bill_dict), len(cp_dict)))
    for root, _, files in os.walk(data):
        if "votes" in root and "data.json" in files:
            congress_num = root.split(os.sep)[1]
            json_data = json.loads(open(root + "/data.json").read())
            if "passage" in json_data["category"] and "bill" in json_data:
                billno = bill_dict[congress_num + json_data["bill"]["type"] + str(json_data["bill"]["number"])]
                for category in json_data["votes"]:
                    if category == "Aye" or category == "Yea":
                        for entry in json_data["votes"][category]:
                            try:
                                vote_matrix[billno,cp_dict[entry["id"]]] = AYE
                            except:
                                print entry
                    elif category == "Nay" or category == "No":
                        for entry in json_data["votes"][category]:
                            vote_matrix[billno,cp_dict[entry["id"]]] = NAY
                    elif category == "Present" or category == "Not Voting":
                        for entry in json_data["votes"][category]:
                            vote_matrix[billno,cp_dict[entry["id"]]] = PRES
    return vote_matrix

def word_valid(word):
    return word.isalpha()

def word_preprocess(word):
    word = filter(lambda x: x.isalnum(), word).lower()
    if word.isdigit():
        return "NUMBER"
    else:
        return word

NUM_WORDS = 1000
def gen_word_dict(data, bill_dict):
    print("Generating word_dict")
    word_dict = {}
    done_bill_dict = {}
    for root, _, files in os.walk(data):
        if "document.txt" in files and any(os.sep + x[:3] + os.sep in root and (os.sep + x[3:] + os.sep in root) and x not in done_bill_dict for x in bill_dict.keys()):
            for bill in bill_dict.keys():
                if os.sep + bill[:3] + os.sep in root and os.sep + bill[3:] + os.sep in root:
                    done_bill_dict[bill] = 1
            bill_file = open(root + "/document.txt").read()
            for word in re.split('\W+', bill_file):
                word = word_preprocess(word)
                if word_valid(word):
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
    sorted_word_list = sorted(word_dict.keys(), key=lambda x: word_dict[x], reverse=True)
    sorted_word_list = sorted_word_list[:NUM_WORDS]
    ret_word_dict = {}
    with open("words.txt", 'w') as outfile:
        for i, entry in enumerate(sorted_word_list):
            ret_word_dict[entry] = i
            outfile.write(entry + '\n')
    return ret_word_dict

def parse_embeddings(pretrained_embed, word_dict):
    print("Parsing Embedding Matrix")
    embeddings = np.zeros((len(word_dict), 50))
    included_words = {}
    with gzip.open(pretrained_embed, 'r') as f:
        content = f.read().split('\n')
        for line in content:
            data = line.split(' ')
            if data[0] in word_dict:
                embeddings[word_dict[data[0]]-1] = map(float, data[1:])
                included_words[data[0]] = 1
    for word in word_dict:
        if word not in included_words:
            # print word
            embeddings[word_dict[word]-1] = np.random.random(50)
    return np.array(embeddings, dtype=np.float)

def gen_doc_term_matrix(data, bill_dict, word_dict):
    print("Generating document-term matrix")
    doc_term_matrix = np.zeros((len(bill_dict), len(word_dict)))
    done_bill_dict = {}
    for root, _, files in os.walk(data):
        if "document.txt" in files and any((os.sep + x[:3] + os.sep in root) and (os.sep + x[3:] + os.sep in root) and x not in done_bill_dict for x in bill_dict.keys()):
            bill_id = -1
            for bill in bill_dict.keys():
                if os.sep + bill[:3] + os.sep in root and os.sep + bill[3:] + os.sep in root:
                    done_bill_dict[bill] = 1
                    bill_id = bill_dict[bill]
            bill_file = open(root + "/document.txt").read()
            for word in bill_file.split():
                word = word_preprocess(word)
                if word_valid(word) and word in word_dict:
                    doc_term_matrix[bill_id, word_dict[word]] = 1
    return doc_term_matrix

def make_party_name_map(cp_dict):
    print "Making name/party dict"
    id_map = {}
    with open('legislators-historic.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['bioguide_id'] in cp_dict:
                id_map[cp_dict[row['bioguide_id']]] = (row['first_name'] + row['last_name'], row['party'])
            if row['lis_id'] in cp_dict:
                id_map[cp_dict[row['lis_id']]] = (row['first_name'] + row['last_name'], row['party'])
    with open('legislators-current.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['bioguide_id'] in cp_dict:
                id_map[cp_dict[row['bioguide_id']]] = (row['first_name'] + row['last_name'], row['party'])
            if row['lis_id'] in cp_dict:
                id_map[cp_dict[row['lis_id']]] = (row['first_name'] + row['last_name'], row['party'])
    cp_info = [id_map[i] for i in range(1 + max(id_map.keys()))]
    f = open("cp_info.txt", "w")
    for (name, party) in cp_info:
        f.write(name + " " + party + "\n")
    return

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    data = dataset

    congressperson_dict, bill_dict = gen_congressperson_bill_dict(data)
    num_bills = len(bill_dict)
    num_cp = len(congressperson_dict)

    word_dict = gen_word_dict(data, bill_dict)
    doc_term_matrix = gen_doc_term_matrix(data, bill_dict, word_dict)
    embedding_matrix = parse_embeddings("glove.6B.50d.txt.gz", word_dict)

    vote_matrix = gen_vote_matrix(data, congressperson_dict, bill_dict)

    make_party_name_map(congressperson_dict)

    big_matrix_train_in = np.zeros((5, (4*num_bills)/5 + 1, NUM_WORDS))
    big_matrix_test_in = np.zeros((5, num_bills/5 + 1, NUM_WORDS))
    big_matrix_train_out = np.zeros((5, (4*num_bills)/5 + 1, num_cp))
    big_matrix_test_out = np.zeros((5, num_bills/5 + 1, num_cp))

    i = 0
    for train, test in KFold(len(vote_matrix), n_folds=5, shuffle=True):
        big_matrix_train_in[i,:len(train),:] = doc_term_matrix[train]
        big_matrix_test_in[i:,:len(test),:] = doc_term_matrix[test]
        big_matrix_train_out[i,:len(train),:] = vote_matrix[train]
        big_matrix_test_out[i,:len(test),:] = vote_matrix[test]
        i += 1

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['embedding_matrix'] = embedding_matrix
        f['big_matrix_train_in'] = big_matrix_train_in
        f['big_matrix_train_out'] = big_matrix_train_out
        f['big_matrix_test_in'] = big_matrix_test_in
        f['big_matrix_test_out'] = big_matrix_test_out
        f['num_bills'] = np.array([len(bill_dict)], dtype=np.int32)
        f['num_cp'] = np.array([len(congressperson_dict)], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
