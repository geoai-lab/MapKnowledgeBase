import pandas as pd
import geopandas as gpd


# Write matched result into a file named by name of matching method.
def write_result_file(df_result, method, text_result_path):
    # Alignments found with text label match will also be added into the final result.
    df_text_matched = pd.read_csv(text_result_path, header=None, sep='\t')
    df_text_matched.columns = ['sou_id', 'tar_id']
    df_result = df_result.append(df_text_matched, ignore_index=True)

    file = method+'.txt'
    df_result.to_csv(file, header=0, index=0, sep='\t')

    return file


# Select the ground truth with labels
def select_ground_truth_labels(entity_set1, entity_set2, ground_truth_path):
    entity_set1_concated = gpd.GeoDataFrame(pd.concat(entity_set1, ignore_index=True))
    entity_set2_concated = gpd.GeoDataFrame(pd.concat(entity_set2, ignore_index=True))
    df_ground_truth = pd.read_csv(ground_truth_path, header=None, sep='\t')
    df_ground_truth.columns = ['sou_id', 'tar_id']

    df_gt_label = pd.DataFrame(columns=('sou_id', 'tar_id'))
    for index, row in df_ground_truth.iterrows():
        sou_label = entity_set1_concated[entity_set1_concated['FeaID'] == row.sou_id].iloc[0]['Label']
        tar_label = entity_set2_concated[entity_set2_concated['FeaID'] == row.tar_id].iloc[0]['Label']
        if sou_label is not None and tar_label is not None:
            df_gt_label = df_gt_label.append(row, ignore_index=True)

    file = 'ground_truth_label.txt'
    df_gt_label.to_csv(file, header=0, index=0, sep='\t')

    return file


# Evaluate the performance based on the matched result and ground truth
def eval_perf(matched_result_path, ground_truth_path):
    df_matched = pd.read_csv(matched_result_path, header=None, sep='\t')
    df_ground_truth = pd.read_csv(ground_truth_path, header=None, sep='\t')
    intersection = pd.merge(df_matched, df_ground_truth, how='inner')
    num_positive = len(intersection)

    precision = num_positive/len(df_matched)
    recall = num_positive/(len(df_ground_truth))
    f1 = 2.0 * ((precision * recall) / (precision + recall))

    print("%s\t%s\t%s" % (round(precision, 4), round(recall, 4), round(f1, 4)))

    return precision, recall, f1