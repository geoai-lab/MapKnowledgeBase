import csv
import os
import pandas as pd
import geopandas as gpd
import re
from evaluate_performance import eval_perf


# Find all entities which have textual labels,and organize them into pairs of the same geometry type.
def generate_pairs_with_label(entity_set1, entity_set2):
    pairs_with_label = []
    entity_set1 = gpd.GeoDataFrame(pd.concat(entity_set1, ignore_index=True))
    entity_set2 = gpd.GeoDataFrame(pd.concat(entity_set2, ignore_index=True))
    for index1, row1 in entity_set1.iterrows():
        geometry1 = row1['geometry']
        for index2, row2 in entity_set2.iterrows():
            geometry2 = row2['geometry']
            if geometry1.geom_type == geometry2.geom_type and str(row1['Label']) != 'None' and str(row2['Label']) != 'None':
                pairs_with_label.append((row1['FeaID'], row1['Label'], row2['FeaID'], row2['Label']))

    return pairs_with_label


# Align entities with labels based on simple string match
def simple_str(label_pairs):
    pairs_string_match = []
    for item in label_pairs:
        if item[1] == item[3]:
            pairs_string_match.append((item[0], item[2]))
    return pairs_string_match


# Align entities with labels based on simple string match after case conversion
def simple_str_case(label_pairs):
    pairs_string_match = []
    for item in label_pairs:
        label1 = item[1].upper()  # Convert all the words into upper case
        label2 = item[3].upper()
        if label1 == label2:
            pairs_string_match.append((item[0], item[2]))

    return pairs_string_match


# Align entities with labels based on simple string match after case conversion and removing the punctuation marks
def simple_str_case_punc(label_pairs):
    rm_list = ['.', ',', '?', '!', ';', '\'', '-', ':', '"', '–']  # top 10 punctuation marks are checked.
    pairs_string_match = []
    for item in label_pairs:
        label1 = item[1].upper()
        label2 = item[3].upper()
        # Remove the punctuation marks existed in textual labels of entities
        for term in rm_list:
            if term in label1:
                label1 = label1.replace(term, '').strip()
        for term in rm_list:
            if term in label2:
                label2 = label2.replace(term, '').strip()

        if label1 == label2:
            pairs_string_match.append((item[0], item[2]))

    return pairs_string_match


# Align entities with labels based on simple string match after case conversion and removing the punctuation marks
# and non-core words
def simple_str_case_punc_noncore(label_pairs):
    rm_list = ['.', ',', '?', '!', ';', '\'', '-', ':', '"', '–', 'AV.', 'PL.', 'ST.', 'AVENUE', 'STREET', 'HALL',
               'BUILDING', 'TOWER', 'ROAD']  # punctuation marks and non-core words need to be checked.

    pairs_string_match = []

    for item in label_pairs:
        label1 = item[1].upper()
        label2 = item[3].upper()
        # Remove the punctuation marks and non-core words existed in textual labels of entities
        for term in rm_list:
            if term in label1:
                label1 = label1.replace(term, '').strip()
        for term in rm_list:
            if term in label2:
                label2 = label2.replace(term, '').strip()

        if label1 == label2:
            pairs_string_match.append((item[0], item[2]))

    return pairs_string_match


# Align entities with labels based on simple string match with case conversion and integrating some domain knowledge
def simple_str_case_punc_noncore_dk(label_pairs):
    rm_list = ['.', ',', '?', '!', ';', '\'', '-', ':', '"', '–', 'AV.', 'PL.', 'ST.', 'AVENUE', 'STREET', 'HALL',
               'BUILDING', 'TOWER', 'ROAD']

    bk_dict = {'1ST': 'FIRST', '2ND': 'SECOND', '3RD': 'THIRD', '4TH': 'FOURTH', '5TH': 'FIFTH', '6TH': 'SIXTH',
               '7TH': 'SEVENTH', '8TH': 'EIGHTH', '9TH': 'NINTH', '10TH': 'TENTH', '11TH': 'ELEVENTH',
               '12TH': 'TWELFTH', '13TH': 'THIRTEENTH', '14TH': 'FOURTEENTH', '15TH': 'FIFTEENTH', '16TH': 'SIXTEENTH',
               '17TH': 'SEVENTEENTH', '18TH': 'EIGHTEENTH', '19TH': 'NINTEENTH', '20TH': 'TWENTIETH'}

    pairs_string_match = []
    for item in label_pairs:
        label1 = item[1].upper()
        label2 = item[3].upper()

        for term in rm_list:
            if term in label1:
                label1 = label1.replace(term, '').strip()
        for term in rm_list:
            if term in label2:
                label2 = label2.replace(term, '').strip()

        # Solve the difference due to different forms of sequence
        for key in bk_dict.keys():
            # use regular expression to make sure that only whole word are matched.
            raw_search_string = r"\b" + key + r"\b"
            matched_string = re.search(raw_search_string, label1)
            if matched_string is not None:
                label1 = label1.replace(key, bk_dict[key])
            matched_string = re.search(raw_search_string, label2)
            if matched_string is not None:
                label2 = label2.replace(key, bk_dict[key])

        if label1 == label2:
            pairs_string_match.append((item[0], item[2]))

    return pairs_string_match


# Combine the results of three machine learning methods.
def ensemble_learning(results_file_path, ground_truth_path):
    combined_text_result = []
    all_text_result = []
    files = os.listdir(results_file_path)
    num_files = len(files)

    # Integrate all the results of three machine methods
    for file in files:
        with open(results_file_path+'/'+file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["feaID1", "feaID2"], delimiter='\t')
            for row in reader:
                all_text_result.append((row["feaID1"], row["feaID2"]))

    # Compute the number of occurrences of each alignment
    counts = pd.value_counts(all_text_result)

    for item, value in counts.items():
        if value == num_files:
            combined_text_result.append(item)

    file = open('combined_text_result.txt', 'w')
    for item in combined_text_result:
        file.write(item[0]+'\t'+item[1]+'\n')
    file.close()

    eval_perf(file.name, ground_truth_path)


# Write the aligned result using textual labels into a file
def labels_result_file(pairs_string_match, method):
    result_file = open(str(method)+'.txt', 'w')
    for item in pairs_string_match:
        result_file.write(item[0])
        result_file.write('\t')
        result_file.write(item[1])
        result_file.write('\n')
    result_file.close()

    return result_file.name