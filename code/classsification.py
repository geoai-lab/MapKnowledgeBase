import pandas as pd

# df_all_matching is used to store all the found alignments in a iteration in the method 'topo'.
df_all_matching = pd.DataFrame(columns=('sou_id', 'tar_id'))


# This function is to implement method of 'topo' iteratively.
def topo(df_similarity, df_matched):
    global df_all_matching

    df_matched.columns = ['sou_id', 'tar_id']

    # Find new matching pairs in one run
    df_new_matching = one_run(df_similarity, df_matched)

    # If there are no new matching pairs found, the result will be evaluated and the method will be done. If there are
    # new matching pairs found, continue to iterate the program.
    if len(df_new_matching) == 0:
        print('done')
    else:
        df_all_matching = df_all_matching.append(df_new_matching, ignore_index=True)
        topo(df_similarity, df_all_matching)

    return df_all_matching


# The method of "topo" align entities is implemented in a iterative manner. Each run will find new alignments.
def one_run(df_similarity, df_matched):
    df_new_matching = pd.DataFrame(columns=('sou_id', 'tar_id'))

    # If all the INNs of two entities have been matched, they will be matching pair.
    for sou_id, group in df_similarity.groupby(['sou_id']):
        for index, row in group.iterrows():
            sou_id = row['sou_id']
            sou_inns = row['topo_sou_inns']
            tar_id = row['tar_id']
            tar_inns = row['topo_tar_inns']

            sum_inns = len(sou_inns) + len(tar_inns)  # the total number of INNs of two entities.

            # Generate all the possible matching pairs with all the INNs
            possilbe_match_list = []
            for m in range(len(sou_inns)):
                current_sou_inn = sou_inns[m]
                for n in range(len(tar_inns)):
                    current_tar_inn = tar_inns[n]
                    possilbe_match_list.append((current_sou_inn, current_tar_inn))

            # Find already matched pairs.
            possible_match_set = set(possilbe_match_list)
            matched_set = set([tuple(value) for value in df_matched.values])
            matched_inns = possible_match_set.intersection(matched_set)

            if (2 * len(matched_inns)) == sum_inns:
                df_new_matching = df_new_matching.append({'sou_id': sou_id, 'tar_id': tar_id}, ignore_index=True)
                break

    return df_new_matching


# This function is to classify entity pairs with the method of 'dist'.
def dist(df_similarity, distance_type):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))
    df_similarity_group = df_similarity.groupby(['sou_id'])
    df_similarity_sorted = df_similarity_group.apply(lambda x: x.sort_values([distance_type]))
    df_similarity_sorted = df_similarity_sorted.rename(columns={'sou_id': 'sou_id1'})

    # Align entities by group. Each group includes the distances between source entity from one map and all entities
    # from another map. If there is only one entity from another map which has shortest distance with source entity,
    # these two entity will be matched.
    for sou_id, group in df_similarity_sorted.groupby(['sou_id1']):
        # For entities of polyline geometry, the angle between entity pairs will also be checked.
        if sou_id.count('line'):
            df_shortest = group[(group[distance_type] == group.iloc[0][distance_type]) & (group['angle'] < 45)][['sou_id1', 'tar_id']]
            df_shortest.columns = ['sou_id', 'tar_id']
            if len(df_shortest) == 1:
                df_result = df_result.append(df_shortest[['sou_id', 'tar_id']], ignore_index=True)
        else:
            df_shortest = group[group[distance_type] == group.iloc[0][distance_type]][['sou_id1', 'tar_id']]
            df_shortest.columns = ['sou_id', 'tar_id']
            if len(df_shortest) == 1:
                df_result = df_result.append(df_shortest[['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to classify entity pairs with the method of 'approx'.
def approx(df_similarity):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))

    # If there is only one target entity which has the relation of 'atr_within' with the source entity, they will be
    # matched.
    df_similarity_group = df_similarity.groupby(['sou_id'])
    df_similarity_sorted = df_similarity_group.apply(lambda x: x.sort_values(['atr_within'], ascending=False))
    df_similarity_sorted = df_similarity_sorted.rename(columns={'sou_id': 'sou_id1'})

    for sou_id, group in df_similarity_sorted.groupby(['sou_id1']):
        df_approx = group[group['atr_within'] >= 0.8]
        df_approx = df_approx.rename(columns={'sou_id1': 'sou_id'})
        if len(df_approx) > 0:
            if len(df_approx) == 1:
                df_result = df_result.append(df_approx[['sou_id', 'tar_id']], ignore_index=True)
            else:
                df_result = df_result.append(df_approx.iloc[0][['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to obtain the result of best distance metric, and this result will be refined with other similarity
# metrics.
def best_dist(df_similarity, distance_method):
    df_similarity_group = df_similarity.groupby(['sou_id'])
    df_similarity_sorted = df_similarity_group.apply(lambda x: x.sort_values([distance_method]))
    df_similarity_sorted = df_similarity_sorted.rename(columns={'sou_id': 'sou_id1'})

    df_result = pd.DataFrame(columns=df_similarity.columns)

    # We will keep all entities which has the shortest distance with source entity.
    for sou_id, group in df_similarity_sorted.groupby(['sou_id1']):
        # For entities of polyline geometry, the angle between entity pairs will be checked.
        if sou_id.count('line'):
            df_shortest = group[(group[distance_method] == group.iloc[0][distance_method]) & (group['angle'] < 45)]
            df_shortest = df_shortest.rename(columns={'sou_id1': 'sou_id'})
            if df_shortest.empty:
                continue
            else:
                df_result = df_result.append(df_shortest, ignore_index=True)
        else:
            df_shortest = group[group[distance_method] == group.iloc[0][distance_method]]
            df_shortest = df_shortest.rename(columns={'sou_id1': 'sou_id'})
            df_result = df_result.append(df_shortest, ignore_index=True)
    return df_result


# This function is to classify entity pairs with the method of 'dist_approx'.
def dist_approx(df_similarity, distance_method):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))

    # Obtain the result of distance-based method.
    df_dist_result = best_dist(df_similarity, distance_method)

    # Refine the obtained result with approximate topological relation.
    df_dist_approx = df_dist_result[df_dist_result['atr_within'] >= 0.8]
    for sou_id, group in df_dist_approx.groupby(['sou_id']):
        if len(group) == 1:
            df_result = df_result.append(group[['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to classify entity pairs with the method of 'dist_topo'.
def dist_topo(df_similarity, text_result_file, distance_method):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))

    df_dist_result = best_dist(df_similarity, distance_method)

    # First check whether there is at least one alignment in the possible entity pairs of INNs of source and target
    # entities. Then, refine the result of distance-based method with 'topo'.
    df_dist_result['topo'] = df_dist_result.apply(refine_topo, args=(text_result_file,), axis=1)
    df_dist_topo = df_dist_result[df_dist_result['topo'] == True]
    for sou_id, group in df_dist_topo.groupby(['sou_id']):
        if len(group) == 1:
            df_result = df_result.append(group[['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to classify entity pairs with the method of 'approx_topo'.
def approx_topo(df_similarity, text_result_file):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))

    # Obtain the result of method 'approx'.
    df_approx = df_similarity[df_similarity['atr_within'] >= 0.8]

    # Refine the result of method 'approx' with 'topo'.
    df_approx['topo'] = df_approx.apply(refine_topo, args=(text_result_file,), axis=1)
    df_approx_topo = df_approx[df_approx['topo'] == True]

    df_approx_topo_group = df_approx_topo.groupby(['sou_id'])
    df_approx_topo_sorted = df_approx_topo_group.apply(lambda x: x.sort_values(['atr_within'], ascending=False))
    df_approx_topo_sorted = df_approx_topo_sorted.rename(columns={'sou_id': 'sou_id1'})

    for sou_id, group in df_approx_topo_sorted.groupby(['sou_id']):
        group = group.rename(columns={'sou_id1': 'sou_id'})
        if len(group) == 1:
            df_result = df_result.append(group[['sou_id', 'tar_id']], ignore_index=True)
        else:
            df_result = df_result.append(group.iloc[0][['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to classify entity pairs with the method of 'dist_topo_approx'.
def dist_topo_approx(df_similarity, text_result_file, distance_method):
    df_result = pd.DataFrame(columns=('sou_id', 'tar_id'))

    df_dist_result = best_dist(df_similarity, distance_method)

    # Refine the result of distance-based method with 'approx' with 'topo'.
    df_dist_result['topo'] = df_dist_result.apply(refine_topo, args=(text_result_file,), axis=1)
    df_dist_approx_topo = df_dist_result[(df_dist_result['atr_within'] >= 0.8) & (df_dist_result['topo'] == True)]
    for sou_id, group in df_dist_approx_topo.groupby(['sou_id']):
        if len(group) == 1:
            df_result = df_result.append(group[['sou_id', 'tar_id']], ignore_index=True)

    return df_result


# This function is to check whether there is at least one alignment in the INNs of source and target entities.
def refine_topo(row, text_result_file):
    df_text_matched = pd.read_csv(text_result_file, header=None, sep='\t')
    df_text_matched.columns = ['sou_id', 'tar_id']

    # Find the INNs of entities
    sou_inns = row['topo_sou_inns']
    tar_inns = row['topo_tar_inns']

    # Generate all the possible matching pairs with all the immediate surrounding entities
    possilbe_match_list = []
    for m in range(len(sou_inns)):
        current_sou_inns = sou_inns[m]
        for n in range(len(tar_inns)):
            current_tar_inns = tar_inns[n]
            possilbe_match_list.append((current_sou_inns, current_tar_inns))

    # Find already matched pairs existing in the result of textual label match.
    possible_match_set = set(possilbe_match_list)
    matched_set = set([tuple(value) for value in df_text_matched.values])
    matched_inns = possible_match_set.intersection(matched_set)

    if len(matched_inns) == 0:
        return False
    else:
        return True