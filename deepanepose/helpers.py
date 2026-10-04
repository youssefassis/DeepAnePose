import json, torch, math
import numpy as np    
from nibabel.affines import apply_affine

def to_tuple( value, length: int = 1):
    """
    to_tuple(1, length=1) -> (1,)
    to_tuple(1, length=3) -> (1, 1, 1)
    If value is an iterable, n is ignored and tuple(value) is returned
    to_tuple((1,), length=1) -> (1,)
    to_tuple((1, 2), length=1) -> (1, 2)
    to_tuple([1, 2], length=3) -> (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value
    
def to_array(value, length=1):
    '''
    This function takes a value and an optional length argument and 
    returns an array of the specified length. It does this by first
    converting the value to a tuple of the specified length using 
    the to_tuple function.
    '''
    return np.array(to_tuple(value, length=length))

def angle_between_vectors(A, B):
    '''
    This function calculates the angle between two vectors in degrees.
    It takes two numpy arrays representing the vectors as input, calculates the 
    dot product between them and divides it by the product of their magnitudes 
    to obtain the cosine of the angle between them. It then uses the inverse 
    cosine (arccos) function from the math module to obtain the angle in radians,
    which is then converted to degrees and returned as the output.
    '''
    cos_angle = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return math.degrees(math.acos(np.clip(cos_angle, -1, 1))) # rounding can push the cosine past 1

def cells_to_spheres(predictions, obj_thresh, input_shape):
    '''
    This function transforms the predicted spheres from the scale of the grid 
    (e.g. 12x12x12) to the patch scale (e.g. 96x96x96).
    Parameters:
        - Predictions: as a np.array
        - obj_thresh: threshold value for the object confidence score.
        - input_shape: as a list
        
    '''
    spheres = []
    predictions = predictions if type(predictions) is np.ndarray else predictions.numpy()
    grid_h, grid_w, grid_d = predictions.shape[0:3] #if nb_anchors == 1 else predictions.shape[1:4]

    predictions[..., -1:] = torch.sigmoid(torch.from_numpy(predictions[..., -1:])).numpy() # normalize confidence scores [0,1]
    predictions[..., 0:3] = torch.sigmoid(torch.from_numpy(predictions[..., 0:3])).numpy() # x y z 's center coords [0, 1]

    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(grid_d):
                confidence = float(predictions[i, j, k, -1])
                if confidence >= obj_thresh:
                    x, y, z, w, h, d = predictions[i, j, k, :-1]
                    # Center
                    x = (i + x) * (input_shape[0] / grid_h)
                    y = (j + y) * (input_shape[1] / grid_w)
                    z = (k + z) * (input_shape[2] / grid_d)
                    # Difference
                    w = w * (input_shape[0] / grid_w)
                    h = h * (input_shape[0] / grid_h)
                    d = d * (input_shape[0] / grid_d)
                    p_x, p_y, p_z = x-w, y-h, z-d
                    spheres.append([x, y, z, p_x, p_y, p_z, confidence])
    return spheres

def non_max_suppression(bboxes, iou_threshold):
    '''
    This function performs non-maximum suppression (NMS) on a list of bounding spheres.
    NMS is a technique used in object detection to filter out overlapping detections of 
    the same object. The function takes in two arguments: bboxes, which is a list of 
    predictions, and iou_threshold, which is the threshold for the intersection over 
    union (IoU) metric.


    '''
    assert type(bboxes) == list
    if len(bboxes) == 0: return bboxes

    bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)    # sort by confidence score
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        results = []
        for box in bboxes:
            radius1 = np.linalg.norm(np.array(box[0:3]) - np.array(box[3:6]))
            radius2 = np.linalg.norm(np.array(chosen_box[0:3]) - np.array(chosen_box[3:6]))
            center1, center2 = box[0:3], chosen_box[0:3]
                        
            if intersection_over_union( np.array(center2+[radius2]), np.array(center1+[radius1])) < iou_threshold:
                results.append(box)
        bboxes = results[:]
        bboxes_after_nms.append(chosen_box)
    return sorted(bboxes_after_nms, key=lambda x: x[-1], reverse=True)
    
def intersection_over_union(prediction, truth):
    '''
    computes the intersection-over-union volume between two spheres
    '''
    def get_intersection(point1, point2):
        pos1, pos2 = point1[:3], point2[:3]
        r1, r2 = point1[3], point2[3]
        d = sum((pos1 - pos2) ** 2) ** 0.5
        
        # check there is no overlapping
        if d >= (r1 + r2):
            vol = 0.
        # check if one sphere entirely holds the other
        elif d == 0 or r1 > (d + r2) or r2 > (d + r1):
            vol = 4. / 3. * np.pi * min(r1, r2) ** 3
        # Partial overlapping
        else:
            vol = (np.pi * (r1 + r2 - d) ** 2 * (d**2 + ((2*d*r1 - 3*r1**2) + (2*d*r2-3 * r2**2)) + 6*r1*r2)) / (12*d)
        return vol
    inter = get_intersection(prediction, truth)
    union = ((4/3) * np.pi * (prediction[3]**3) +  (4/3) * np.pi * (truth[3]**3)) -  inter
    return inter / (union + 1e-10)

def flip_coords (spheres, axis, dimension):
    '''
    This function flips the coordinates of a list of spheres along a given axis.
    It takes in three arguments: 
        - spheres: a list of sphere coordinates,
        - axis: an integer indicating the axis to flip along,
        - dimension: a list of integers representing the size of each dimension 
        of the space the spheres are in.
    '''
    for sphere in spheres:
        sphere[axis] = abs (dimension[axis] - 1 - sphere[axis])
        if len(sphere)>4:
            sphere[axis+3] = abs (dimension[axis] - 1 - sphere[axis+3])
    return spheres

def save_json(data: tuple, path: str, indent: int = 4):
    '''
    This function saves a tuple of data as a JSON file to a specified file path. 
    '''
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

def save_spheres(spheres, out_file):    
    '''
    This is a Python function that saves a list of sphere coordinates as a JSON file 
    to a specified file path. 
    The function takes in two arguments: spheres, which is a list of sphere coordinates,
    and out_file, which is a string representing the file path where the JSON file will
    be saved. The sphere's ostium and dome, center points, radius, and confidence score
    are extracted from the given sphere coordinates and added/saved to the dictionary.
    '''
    data = {}
    if len(spheres)==0:
        save_json(data = data, path = out_file)
        
    for id, cc in enumerate (spheres):
        center, ostium = list(cc[:3]), list(cc[3:6])
        dome = [i + f for i, f in zip(center, [i-j for i, j in zip(center, ostium)])]
        radius = np.linalg.norm(np.array(cc[:3]) - np.array(cc[3:6]))
        data[id] = {"P1": ostium, "P2": dome, "center": center, "radius": float(radius), "confidence": float(cc[-1])}
        save_json(data = data, path = out_file)    
    
def keep_only_intersected_spheres(spheres1, spheres2, min_iou_threshold):
    assert type(spheres1) == type(spheres2) == list
    if len(spheres1) == 0 or len(spheres2) == 0: return []

    results = []
    for sphere1 in spheres1:
        for sphere2 in spheres2:
            radius1 = np.linalg.norm(np.array(sphere1[0:3]) - np.array(sphere1[3:6]))
            radius2 = np.linalg.norm(np.array(sphere2[0:3]) - np.array(sphere2[3:6]))
            
            center1, center2 = sphere1[0:3], sphere2[0:3]

            iou = intersection_over_union( np.array(center2+[radius2]), np.array(center1+[radius1]))
                
            if iou >= min_iou_threshold:
                # take the shere with high confidence + avg (center & radius)
                if sphere1[-1] >= sphere2[-1]:
                    results.append(sphere1)
                else : 
                    results.append(sphere2)
    return non_max_suppression(sorted(results, key=lambda x: x[-1], reverse=True), iou_threshold=min_iou_threshold )

def spheres_to_metric (spheres, v2m):
    '''
    This is a Python function that transforms a list of sphere coordinates 
    from voxel space to metric space using an affine transformation matrix. 
    The function takes in two arguments: spheres, which is a list of sphere 
    coordinates in voxel space, and v2m, which is a 4x4 affine transformation 
    matrix that converts voxel coordinates to metric coordinates.
    '''
    def apply_transformation(elements, trans):
        return apply_affine (trans, elements)
    assert isinstance(spheres, list)
    for sphere in spheres:
        sphere[:3] = apply_transformation(np.array(sphere[:3]), v2m.squeeze())
        sphere[3:6] = apply_transformation(np.array(sphere[3:6]), v2m.squeeze())
    return spheres
