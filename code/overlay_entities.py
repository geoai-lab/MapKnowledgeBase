import arcpy
import geopandas as gpd
import shapely
import csv
import os
import pandas as pd
from shapely.geometry import Point
import re
from shapely.geometry import MultiPoint
import shutil
from shapely.validation import explain_validity


# Affine transformation with the generated control points.
def affine_trans(entity_set1, entity_set2, text_result_file):
    # Compute control points with the result of textual label match.
    df_control_points = generate_control_points(entity_set1, entity_set2, text_result_file)

    # Examine whether rubber sheeting can be performed to further adjust the spatial positions of the entities.
    # This also means whether entities can be overlaid.
    overlaid = True

    # If the number of found control points is less than 3, overlaid will be False and the return will be original
    # entities.
    if len(df_control_points) < 3:
        overlaid = False
        entity_set1 = gpd.GeoDataFrame(pd.concat(entity_set1, ignore_index=True))
        entity_set2 = gpd.GeoDataFrame(pd.concat(entity_set2, ignore_index=True))
        return entity_set1, entity_set2, overlaid

    # If the number of found control points is greater than 3, affine transformation will be performed. In order to
    # remove potentially wrong found control points, this process includes three steps.
    # The first step is to perform an initial affine transformation.
    links_sour_tar(df_control_points)  # Generate map links using all found control points.
    folder_affine_trans('affine_trans1')  # New a folder to store the result of affine transformation.
    trans_entity_set1 = []
    # Each shapefile of the first dataset will be transformed.
    for i in range(len(entity_set1)):
        # Copy the shapefile to be transformed.
        locals()['gdf_' + str(i)] = entity_set1[i].copy()
        locals().get('gdf_' + str(i)).to_file('affine_trans1/gdf_' + str(i) + '.shp')
        # Perform affine transformation.
        arcpy.TransformFeatures_edit(in_features='affine_trans1/gdf_' + str(i) + '.shp', in_link_features="links.shp", method='AFFINE')
        trans_entity_set1.append(gpd.read_file('affine_trans1/gdf_' + str(i) + '.shp'))

    # Filter control points based on spatial distances of the control points computed with the transformed result.
    trans_entity_set2 = entity_set2
    df_control_points_filtered = filter_cp(trans_entity_set1, trans_entity_set2, df_control_points)

    # Affine transformation again with filtered control points.
    links_sour_tar(df_control_points_filtered)
    folder_affine_trans('affine_trans2')
    trans_entity_set1 = []
    for i in range(len(entity_set1)):
        locals()['gdf_' + str(i)] = entity_set1[i].copy()
        locals().get('gdf_' + str(i)).to_file('affine_trans2/gdf_' + str(i) + '.shp')
        arcpy.TransformFeatures_edit(in_features='affine_trans2/gdf_' + str(i) + '.shp',
                                       in_link_features="links.shp", method='AFFINE')
        trans_entity_set1.append(gpd.read_file('affine_trans2/gdf_' + str(i) + '.shp'))

    return trans_entity_set1, trans_entity_set2, overlaid


# Compute control points based on the matched entities with labels.
# There are two types of control points: (1) entities with point geometry; (2) the same intersections of roads.
def generate_control_points(entity_set1, entity_set2, text_result_file):
    entity_set1 = gpd.GeoDataFrame(pd.concat(entity_set1, ignore_index=True))
    entity_set2 = gpd.GeoDataFrame(pd.concat(entity_set2, ignore_index=True))
    df_control_points = pd.DataFrame(columns=('feaIDs', 'cp1', 'cp2', 'type'))
    matched_alignments_point = []
    matched_alignments_polyline = []

    # Read matched entity pairs in the geometry of polyline or point.
    with open(text_result_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["feaID1", "feaID2"], delimiter='\t')
        for row in reader:
            a_geometry = entity_set1[entity_set1.FeaID.isin([row['feaID1']])].iloc[0].geometry
            b_geometry = entity_set2[entity_set2.FeaID.isin([row['feaID2']])].iloc[0].geometry
            length = a_geometry.length
            area = a_geometry.area
            if length == 0.0 and area == 0.0:
                matched_alignments_point.append((row["feaID1"], row["feaID2"], a_geometry, b_geometry))
            if length != 0.0 and area == 0.0:
                matched_alignments_polyline.append((row["feaID1"], row["feaID2"], a_geometry, b_geometry))

    # Generate control points based on matched point entities
    for item in matched_alignments_point:
        df_control_points = df_control_points.append([{'feaIDs': item[0] + ' ' + item[1], 'cp1': list(item[2].coords),
                                                       'cp2': list(item[3].coords), 'type': 'point'}], ignore_index=True)

    # Compute control points of matched polyline entities
    for i in range(len(matched_alignments_polyline)):
        item1 = matched_alignments_polyline[i]
        polyline1 = item1[2]
        polyline2 = item1[3]
        for j in range(len(matched_alignments_polyline)):
            item2 = matched_alignments_polyline[j]
            # Duplicate alignments will not be used to compute control points.
            if (i <= j) or (item1[0] == item2[0]) or (item1[1] == item2[1]):
                continue
            else:
                polyline3 = item2[2]
                polyline4 = item2[3]
                intersection1 = polyline1.intersection(polyline3)
                intersection2 = polyline2.intersection(polyline4)
                # Two computed intersections should not be empty.
                if intersection1.is_empty or intersection2.is_empty:
                    continue
                else:
                    df_control_points = df_control_points.append(
                        [{'feaIDs': item1[0] + ' ' + item2[0] + ' ' + item1[1] + ' ' + item2[1],
                          'cp1': list(intersection1.coords), 'cp2': list(intersection2.coords),
                          'type': 'intersection'}], ignore_index=True)

    print('The number of identified control points is %s' % len(df_control_points))

    return df_control_points


# This method is to filter control points based on spatial distance of control points using transformed maps.
def filter_cp(trans_entity_set1, trans_entity_set2, df_control_points):
    trans_entity_set1 = gpd.GeoDataFrame(pd.concat(trans_entity_set1, ignore_index=True))
    trans_entity_set2 = gpd.GeoDataFrame(pd.concat(trans_entity_set2, ignore_index=True))

    df_control_points['distance'] = ''

    # Compute the distance between control points based on the transformed entity sets.
    for index, row in df_control_points.iterrows():
        # Distance between control points based on point entity
        if row['type'] == 'point':
            feaIDs = row['feaIDs'].split(' ')
            point_a = trans_entity_set1[trans_entity_set1.FeaID.isin([feaIDs[0]])].iloc[0].geometry
            point_b = trans_entity_set2[trans_entity_set2.FeaID.isin([feaIDs[1]])].iloc[0].geometry
            distance = point_a.distance(point_b)
            row['distance'] = distance
        # Distance between control points based on polyline entity
        else:
            feaIDs = row['feaIDs'].split(' ')
            line_1a = trans_entity_set1[trans_entity_set1.FeaID.isin([feaIDs[0]])].iloc[0].geometry
            line_1b = trans_entity_set1[trans_entity_set1.FeaID.isin([feaIDs[1]])].iloc[0].geometry
            line_2a = trans_entity_set2[trans_entity_set2.FeaID.isin([feaIDs[2]])].iloc[0].geometry
            line_2b = trans_entity_set2[trans_entity_set2.FeaID.isin([feaIDs[3]])].iloc[0].geometry

            intersection1 = line_1a.intersection(line_1b)
            intersection2 = line_2a.intersection(line_2b)

            distance = intersection1.distance(intersection2)
            row['distance'] = distance

    # The control point pairs whose distances are two standard deviation away from the mean distance will be removed.
    df_control_points['tag'] = 'Y'
    mean_dis = df_control_points['distance'].mean()
    std_dis = df_control_points['distance'].std()

    for index, row in df_control_points.iterrows():
        if abs(row['distance']-mean_dis) > 2*std_dis:
            row['tag'] = 'N'

    df_control_points_filtered = df_control_points[df_control_points['tag'] == 'Y']
    df_control_points_filtered = df_control_points_filtered[['feaIDs', 'cp1', 'cp2', 'type']]

    print('The number of filtered control points is %s' % len(df_control_points_filtered))

    return df_control_points_filtered


# Generate map links using computed control points.Each link refers to a line connected by two control points which
# are from source map and target map respectively and are the same point in the real world.
def links_sour_tar(df_control_points):
    link_geometry = []
    for index, row in df_control_points.iterrows():
        # Obtain the coordinates of control points
        coordinates1 = re.sub('[\[()\]]', '', str(row['cp1'])).split(',')
        coordinates2 = re.sub('[\[()\]]', '', str(row['cp2'])).split(',')

        # Generate links between maps
        control_point1 = Point(float(coordinates1[0]), float(coordinates1[1]))
        control_point2 = Point(float(coordinates2[0]), float(coordinates2[1]))
        link_line = shapely.geometry.LineString([control_point1, control_point2])
        link_geometry.append(link_line)
    links_gdf = gpd.GeoDataFrame(geometry=link_geometry)
    links_gdf.to_file("links.shp")


# For maps which have georeferencing information, if necessary, make the CRSs of maps same by CRS transformation.
def transformation_crs(entity_set2, entity_set_crs):
    entity_set_transformed = []
    for item in entity_set2:
        item.to_crs(entity_set_crs)
        entity_set_transformed.append(item)

    return entity_set_transformed


# Search the entities which are within the overlapping area of two entity sets. Only entities within the overlapping
# area will be processed further.
def overlapping_entity_pairs(entity_set1_overlaid, entity_set2_overlaid):
    entity_set1_overlaid = gpd.GeoDataFrame(pd.concat(entity_set1_overlaid, ignore_index=True))
    entity_set2_overlaid = gpd.GeoDataFrame(pd.concat(entity_set2_overlaid, ignore_index=True))
    all_vertices1 = []
    all_vertices2 = []

    # Obtain vertices of all entities to be matched.
    for index, row in entity_set1_overlaid.iterrows():
        if row.geometry.geom_type == 'Polygon':
            all_vertices1.append(list(row.geometry.exterior.coords))
        else:
            all_vertices1.append(list(row.geometry.coords))
    all_vertices1 = MultiPoint(sum(all_vertices1, []))

    for index, row in entity_set2_overlaid.iterrows():
        if row.geometry.geom_type == 'Polygon':
            all_vertices2.append(list(row.geometry.exterior.coords))
        else:
            all_vertices2.append(list(row.geometry.coords))
    all_vertices2 = MultiPoint(sum(all_vertices2, []))

    # Compute overlapping area by computing the intersection area of convex_hulls which are generated with all vertices.
    all_vertices1_convexhull = all_vertices1.convex_hull
    all_vertices2_convexhull = all_vertices2.convex_hull
    overlapping_area = all_vertices1_convexhull.intersection(all_vertices2_convexhull)
    links_gdf = gpd.GeoDataFrame(geometry=[overlapping_area])
    links_gdf.to_file("intersection.shp")

    # Those entities which do not intersect with the overlapping area will be removed.
    entity_set1_overlaid_copy = entity_set1_overlaid.copy()
    for index, row in entity_set1_overlaid.iterrows():
        if explain_validity(row.geometry) == 'Valid Geometry':
            if row.geometry.intersect(overlapping_area).is_empty:
                delete_index = entity_set1_overlaid_copy[entity_set1_overlaid_copy['FeaID'] == row.FeaID].index.tolist()
                entity_set1_overlaid_copy.drop(entity_set1_overlaid_copy.index[delete_index], inplace=True)
                entity_set1_overlaid_copy.index = range(len(entity_set1_overlaid_copy))

    entity_set2_overlaid_copy = entity_set2_overlaid.copy()
    for index, row in entity_set2_overlaid.iterrows():
        if explain_validity(row.geometry) == 'Valid Geometry':
            if row.geometry.intersect(overlapping_area).is_empty:
                delete_index = entity_set2_overlaid_copy[entity_set2_overlaid_copy['FeaID'] == row.FeaID].index.tolist()
                entity_set2_overlaid_copy.drop(entity_set2_overlaid_copy.index[delete_index], inplace=True)
                entity_set2_overlaid_copy.index = range(len(entity_set2_overlaid_copy))

    return entity_set1_overlaid_copy, entity_set2_overlaid_copy


# New a folder to store result of affine transformation.
def folder_affine_trans(folder_name):
    if folder_name not in os.listdir('./'):
        os.mkdir(folder_name)
    else:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)