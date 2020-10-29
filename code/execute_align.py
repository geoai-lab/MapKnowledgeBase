import geopandas as gpd
import text_label_match
import overlay_entities
import evaluate_performance
import pandas as pd
import similarity_computation
import classsification
import pickle


# This function is the main function of our method. The input of it is two digitized maps, the ground truth, and string
# of text label match method. The output is a .pkl file which stores the computed similarity between entities from two
# input maps. One digitized map may include three vector data files (point, polyline, and polygon), or include part of
# them.
def execute_alignment(shapefile_list1, shapefile_list2, ground_truth, text_label_method, only_text=False):
    entity_set1 = []
    entity_set2 = []
    entity_set_crs_tag = True

    # Obtain entities set from ShapeFiles, and examine whether they have georeferencing information. If both maps have
    # georeferencing information, this program will go to compute overlapping area and similarity directly. Otherwise,
    # these maps will be checked whether affine transformation can be employed on them.
    for i in range(len(shapefile_list1)):
        entity_set = gpd.read_file(shapefile_list1[i])
        if not entity_set.empty:
            entity_set1.append(entity_set)
            if not entity_set.crs:
                entity_set_crs_tag = False
            else:
                entity_set_crs1 = entity_set.crs

    for i in range(len(shapefile_list2)):
        entity_set = gpd.read_file(shapefile_list2[i])
        if not entity_set.empty:
            entity_set2.append(entity_set)
            if not entity_set.crs:
                entity_set_crs_tag = False
            else:
                entity_set_crs2 = entity_set.crs

    # If only_text is True, this function will only retrieve alignments with textual labels.
    if only_text:
        ground_truth_label = evaluate_performance.select_ground_truth_labels(entity_set1, entity_set2, ground_truth)
        textual_label_alignment(entity_set1, entity_set2, text_label_method, ground_truth_label)
    else:
        # If two entity sets have georeference information, perform necessary CRS transformation to make the CRSs of two
        # entity sets same.
        if entity_set_crs_tag:
            # Transform the CRS of entity_set2 to the CRS of entity_set1.
            text_result_file = textual_label_alignment(entity_set1, entity_set2, text_label_method, ground_truth)
            if dict.cmp(entity_set_crs1, entity_set_crs2) == 0:
                entity_set2_overlaid = overlay_entities.transformation_crs(entity_set2, entity_set_crs1)
            entity_set1_overlapping, entity_set2_overlapping = overlay_entities.overlapping_entity_pairs(entity_set1, entity_set2_overlaid)
            similarity_calculation(entity_set1_overlapping, entity_set2_overlapping, text_result_file, True)
        # Compute control points with alignments found with text label match. Then according to the computed control points,
        # whether maps can be transformed and overlaid will be checked.
        else:
            text_result_file = textual_label_alignment(entity_set1, entity_set2, text_label_method, ground_truth)
            trans_entity_set1, trans_entity_set2, overlaid = overlay_entities.affine_trans(entity_set1, entity_set2, text_result_file)
            # If maps are overlaid, we will compute overlapping entities first, and then compute similarity between
            # overlapping entities.
            if overlaid:
                entity_set1_overlapping, entity_set2_overlapping = overlay_entities.overlapping_entity_pairs(trans_entity_set1, trans_entity_set2)
                similarity_calculation(entity_set1_overlapping, entity_set2_overlapping, text_result_file, overlaid)
            # If maps can not be overlaid, compute the similarity of feature 'topo' only.
            else:
                similarity_calculation(trans_entity_set1, trans_entity_set2, text_result_file, overlaid)


# With two input maps, this function is to align entities with textual labels using a certain text label
# match method.
def textual_label_alignment(entity_set1, entity_set2, text_label_method, ground_truth_label):
    # Build textual label pairs of entities with the same type of geometries.
    entity_label_pairs = text_label_match.generate_pairs_with_label(entity_set1, entity_set2)

    # Align entities with selected textual label match method.
    # If one machine learning method is selected, it will directly obtain the result and compute the performance.
    machine_learning_methods = ['text_santos2018b', 'text_santos2018a', 'text_acheson2019', 'ensemble_learning']
    if text_label_method in machine_learning_methods:
        getattr(text_label_match, text_label_method)(entity_label_pairs, ground_truth_label)
    else:
        text_align_result = getattr(text_label_match, text_label_method)(entity_label_pairs)
        text_result_file = text_label_match.labels_result_file(text_align_result, text_label_method)
        evaluate_performance.eval_perf(text_result_file, ground_truth_label)
        return text_result_file


# With the processed entities, this function is to compute similarity depending on the different cases of processing
# entities.
def similarity_calculation(entity_set1_processed, entity_set2_processed, text_result_file, overlaid):
    df_text_matched = pd.read_csv(text_result_file, header=None, sep='\t')
    df_text_matched.columns = ['sou_id', 'tar_id']

    # This piece of code is to build all possible entity pairs of two maps, and those entities which have been matched
    # using textual label match method will not be aligned further.
    df_similarity = pd.DataFrame(columns=('sou_id', 'tar_id', 'sou_feature', 'tar_feature'))
    num = 0
    for index1, item1 in entity_set1_processed.iterrows():
        if len(df_text_matched[df_text_matched['sou_id'].eq(item1['FeaID'])]) != 0:
            continue
        for index2, item2 in entity_set2_processed.iterrows():
            if len(df_text_matched[df_text_matched['tar_id'].eq(item2['FeaID'])]) != 0:
                continue
            if item1.geometry.geom_type == item2.geometry.geom_type:
                df_similarity.loc[num] = [item1['FeaID'], item2['FeaID'], item1.geometry, item2.geometry]
                num = num + 1

    # If two maps can be overlaid, four types of distance, angle of polyline entities, approximate topological
    # relations, and INNs will be computed. Otherwise, only INNs can be computed.
    if overlaid:
        df_similarity['dist_edc'] = df_similarity.apply(similarity_computation.edc, axis=1)
        df_similarity['dist_edv'] = df_similarity.apply(similarity_computation.edv, axis=1)
        df_similarity['dist_hdv'] = df_similarity.apply(similarity_computation.hdv, axis=1)
        df_similarity['dist_ednp'] = df_similarity.apply(similarity_computation.ednp, axis=1)
        df_similarity['angle'] = df_similarity.apply(similarity_computation.angle_lines, axis=1)

        radius = similarity_computation.compute_radius(df_similarity)
        df_similarity['atr_within'] = df_similarity.apply(similarity_computation.atr_within, args=(radius, ), axis=1)

        df_similarity['topo_sou_inns'] = df_similarity.apply(similarity_computation.topo_sou, args=(entity_set1_processed,), axis=1)
        df_similarity['topo_tar_inns'] = df_similarity.apply(similarity_computation.topo_tar, args=(entity_set2_processed,), axis=1)
    else:
        df_similarity['topo_sou_inns'] = df_similarity.apply(similarity_computation.topo_sou, args=(entity_set1_processed,), axis=1)
        df_similarity['topo_tar_inns'] = df_similarity.apply(similarity_computation.topo_tar, args=(entity_set2_processed,), axis=1)

    # The computed dataframe of similarity will be written in a pkl file.
    with open('all.pkl', 'wb') as pickle_file:
        pickle.dump(df_similarity, pickle_file)


# With the pkl file containing the computed similarity scores, this function makes alignment classification.
def alignment_classification(df_similarity_path, method_name, text_result_file, ground_truth, distance_method=None):
    # Read the pkl file of similarity.
    with open(df_similarity_path, 'rb') as pickle_file:
        df_similarity = pickle.load(pickle_file)

    # Corresponding columns of similarity will be chosen according to the name of used classification method.
    method_dict = {'topo': ['topo_sou_inns', 'topo_tar_inns'], 'dist': ['dist_edc', 'dist_edv', 'dist_hdv', 'dist_ednp', 'angle'],
                       'approx': ['atr_within'], 'dist_topo': [distance_method, 'angle', 'topo_sou_inns', 'topo_tar_inns'], 'dist_approx': [distance_method, 'angle', 'atr_within'],
                       'approx_topo': ['atr_within', 'topo_sou_inns', 'topo_tar_inns'],
                       'dist_topo_approx': [distance_method, 'angle', 'topo_sou_inns', 'topo_tar_inns', 'atr_within']}
    selected_columns = sum([['sou_id'], ['tar_id'], method_dict[method_name]], [])
    df_similarity = df_similarity[selected_columns]

    # Different names of classification method will call the corresponding classification function, obtain the result,
    # and evaluate the performance.
    if method_name == 'topo':
        df_matched = pd.read_csv(text_result_file, header=None, sep='\t')
        df_result = getattr(classsification, method_name)(df_similarity, df_matched)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'dist':
        distance_types = ['dist_edc', 'dist_edv', 'dist_ednp', 'dist_hdv']
        for i in range(len(distance_types)):
            df_result = getattr(classsification, method_name)(df_similarity, distance_types[i])
            result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
            evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'approx':
        df_result = getattr(classsification, method_name)(df_similarity)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'dist_topo':
        df_result = getattr(classsification, method_name)(df_similarity, text_result_file, distance_method)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'dist_approx':
        df_result = getattr(classsification, method_name)(df_similarity, distance_method)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'approx_topo':
        df_result = getattr(classsification, method_name)(df_similarity, text_result_file)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)

    if method_name == 'dist_topo_approx':
        df_result = getattr(classsification, method_name)(df_similarity, text_result_file, distance_method)
        result_file = evaluate_performance.write_result_file(df_result, method_name, text_result_file)
        evaluate_performance.eval_perf(result_file, ground_truth)