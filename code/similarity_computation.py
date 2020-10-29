from shapely.geometry import MultiPoint
import numpy as np
from shapely.validation import explain_validity
from shapely.geometry import LineString
from shapely.ops import nearest_points


# Distance between entities of point geometry
def point_distance(point1, point2):
    return point1.distance(point2)


# Euclidean distance between centroids of entities
def edc(row):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    if geometry_sou.geom_type == "Point" and geometry_tar.geom_type == 'Point':
        distance = point_distance(geometry_sou, geometry_tar)
    else:
        distance = geometry_sou.centroid.distance(geometry_tar.centroid)
    return distance


# Shortest euclidean distance between vertices of entities
def edv(row):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    if geometry_sou.geom_type == "Point" and geometry_tar.geom_type == 'Point':
        distance = point_distance(geometry_sou, geometry_tar)
    else:
        # Obtain all the vertices of entities
        if geometry_sou.geom_type == 'Polygon':
            vertices1 = list(geometry_sou.exterior.coords)
        else:
            vertices1 = list(geometry_sou.coords)
        vertices1 = MultiPoint(vertices1)

        if geometry_tar.geom_type == 'Polygon':
            vertices2 = list(geometry_tar.exterior.coords)
        else:
            vertices2 = list(geometry_tar.coords)
        vertices2 = MultiPoint(vertices2)

        distance = vertices1.distance(vertices2)

    return distance


# Hausdorff distance with vertices of entities
def hdv(row):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    if geometry_sou.geom_type == "Point" and geometry_tar.geom_type == 'Point':
        distance = point_distance(geometry_sou, geometry_tar)
    else:
        distance = geometry_sou.hausdorff_distance(geometry_tar)
    return distance


# Euclidean distance of nearest points between entities
def ednp(row):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    if geometry_sou.geom_type == "Point" and geometry_tar.geom_type == 'Point':
        distance = point_distance(geometry_sou, geometry_tar)
    else:
        distance = geometry_sou.distance(geometry_tar)

    return distance


# Compute the angle between entities of polyline
def angle_lines(row):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    if geometry_sou.geom_type == 'LineString' and geometry_tar.geom_type == 'LineString':

        coord_list1 = list(geometry_sou.coords)
        coord_list2 = list(geometry_tar.coords)
        # Vectors of first and second points of entities
        arr_a = np.array([(coord_list1[1][0] - coord_list1[0][0]), (coord_list1[1][1] - coord_list1[0][1])])
        arr_b = np.array([(coord_list2[1][0] - coord_list2[0][0]), (coord_list2[1][1] - coord_list2[0][1])])

        # Cosine between two vectors
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))
        angle = np.arccos(cos_value) * 180 / np.pi
        if angle > 90:
            angle = 180 - angle
        return angle


# Compute radius to be used to generate buffer zones.
def compute_radius(df_distance):
    # Minimum 0.05 quantile of ascending sorted distance matrices will be chosen as the radius.
    nth_distance = int(len(df_distance) * 0.05)-1

    # Compute 0.05 quantile for each type of distances.
    df_distance.sort_values('dist_edc', inplace=True)
    radius_edc = df_distance.iloc[nth_distance]['dist_edc']
    df_distance.sort_values('dist_edv', inplace=True)
    radius_edv = df_distance.iloc[nth_distance]['dist_edv']
    df_distance.sort_values('dist_hdv', inplace=True)
    radius_hdv = df_distance.iloc[nth_distance]['dist_hdv']
    df_distance.sort_values('dist_ednp', inplace=True)
    radius_ednp = df_distance.iloc[nth_distance]['dist_ednp']

    radius = round(min(radius_edc, radius_ednp, radius_edv, radius_hdv), 2)

    return radius


# Examine entity pair to be matched whether they have approximate topological relations of approximately within.
def atr_within(row, radius):
    geometry_sou = row['sou_feature']
    geometry_tar = row['tar_feature']
    buf_1 = geometry_sou.buffer(radius)
    buf_2 = geometry_tar.buffer(radius)

    if explain_validity(geometry_sou) == 'Valid Geometry' and explain_validity(geometry_tar) == 'Valid Geometry':
        area = buf_1.intersection(buf_2).area
        area_ratio = area/(min(buf_1.area, buf_2.area))
        return area_ratio


# Compute INNs for source entity.
def topo_sou(row, entity_set):
    inns = []
    row_sou_geometry = row['sou_feature']
    row_sou_id = row['sou_id']

    # Check each one entity whether it is an INN for the current entity.
    for index, entity in entity_set.iterrows():
        is_immediate = True
        geometry = entity.geometry
        computed_nearest_segments = nearest_segments(row_sou_geometry, geometry)
        if entity['FeaID'] == row_sou_id:
            continue

        # If the generated nearest segment intersects with any other entity except the source entities and current
        # entities, this entity will not be regarded as an INN of source entity.
        for i in range(len(entity_set)):
            current_geometry = entity_set.iloc[i].geometry
            if computed_nearest_segments.intersects(current_geometry) and (
                    (entity_set.iloc[i].FeaID == row['sou_id']) is False) and (
                    (entity_set.iloc[i].FeaID == entity['FeaID']) is False):
                is_immediate = False
                break
        if is_immediate:
            inns.append(entity['FeaID'])

    return inns


# Compute INNs for target entity.
def topo_tar(row, entity_set):
    inns = []
    row_tar_geometry = row['tar_feature']
    row_tar_id = row['tar_id']

    for index, entity in entity_set.iterrows():
        is_immediate = True
        geometry = entity.geometry
        computed_nearest_segments = nearest_segments(row_tar_geometry, geometry)
        if entity['FeaID'] == row_tar_id:
            continue

        for i in range(len(entity_set)):
            current_geometry = entity_set.iloc[i].geometry
            if computed_nearest_segments.intersects(current_geometry) and (
                    (entity_set.iloc[i].FeaID == row['tar_id']) is False) and (
                    (entity_set.iloc[i].FeaID == entity['FeaID']) is False):
                is_immediate = False
                break
        if is_immediate:
            inns.append(entity['FeaID'])

    return inns


# Generate the nearest segments for each two geometries
def nearest_segments(geometry1, geometry2):
    generated_nearest_points = [o for o in nearest_points(geometry1, geometry2)]
    generated_nearest_segments = LineString([generated_nearest_points[0], generated_nearest_points[1]])
    return generated_nearest_segments


