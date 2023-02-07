# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import skimage.measure as measure
import time

def get_current_time():
    return time.strftime('%H:%M:%S')


def get_layer_id_by_name(viewer, nm):
    """Get index of layer in the napari.viewer by its name
    """
    count = 0
    for i in viewer.layers:
        if i.name == nm:
            return count
        count += 1
    return


def get_annotation(track, hasState, stateColName):
    ann = []
    cls_col = stateColName
    track_id = list(track['trackId'])
    parent_id = list(track['parentTrackId'])
    cls_lb = list(track[cls_col])
    for i in range(track.shape[0]):
        if hasState:
            inform = [str(track_id[i]), str(parent_id[i]), cls_lb[i]]
        else:
            inform = [str(track_id[i]), str(parent_id[i])]
        if inform[1] == '0':
            del inform[1]
        ann.append('-'.join(inform))
    track['name'] = ann
    return track


def align_table_and_mask(table, mask, align_morph=False):
    """For every object in the mask, check if is consistent with the table. If no, remove the object in the mask.

    Args:
        table (pandas.DataFrame): (tracked) object table.
        mask (numpy.ndarray): labeled object mask, object label should be corresponding to `continuous_label` column in the table.
        align_morph (bool): align morphologically (match xy coordinate) or not.
    """

    count = 0
    count_up = 0
    if align_morph:
        new = pd.DataFrame()
    for i in range(mask.shape[0]):
        sub = table[table['frame'] == i].copy()
        sls = mask[i,:,:].copy()
        lbs = sorted(list(np.unique(sls)))
        if lbs[0] == 0:
            del lbs[0]
        registered = list(sub['continuous_label'])
        rmd = list(set(lbs) - set(registered))
        if rmd:
            for j in rmd:
                sls[sls == j] = 0
                count += 1
            mask[i,:,:] = sls
        if align_morph:
            props = measure.regionprops(mask[i,:,:])
            for p in props:
                lb = p.label
                obj = sub[sub['continuous_label'] == lb]
                if obj.shape[0]<1:
                    raise ValueError('Object in the mask not registered in the table!')
                y,x = p.centroid
                if np.round(obj['Center_of_the_object_0'].iloc[0],3) == np.round(x,3) and np.round(obj['Center_of_the_object_1'].iloc[0],3) == np.round(y,3):
                    # The object is unchanged if coordinate matches
                    continue
                else:
                    print('Update object ' + str(lb) + ' at frame ' + str(i))
                    count_up += 1
                    # Update morphology
                    sub.loc[obj.index, 'Center_of_the_object_0'] = x
                    sub.loc[obj.index, 'Center_of_the_object_1'] = y
            new = pd.concat([new, sub.copy()])
    
    print('Removed ' + str(count) + ' objects.')

    if align_morph:
        print('Updated ' + str(count_up) + ' objects.')
        return mask, new
    else:
        return mask


def expand_bbox(bbox, factor, limit):
    """Expand bounding box by factor times.

    Args:
        bbox (tuple): (x1, y1, x2, y2).
        factor (float): positive value, expand height and width by multiplying the factor.
            Round if result is not integer.
            The output shape will be (factor + 1) ** 2 times of the original size.
        limit (tuple): (x_max, y_max), limit values to avoid boundary crush.

    Returns:
        (tuple): new bounding box (x1, y1, x2, y2).
    """
    if factor < 0:
        raise ValueError('Must expand bounding box with a positive factor.')

    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    factor = factor / 2
    x1, y1, x2, y2 = bbox
    x1 -= factor * h
    y1 -= factor * w
    x2 += factor * h
    y2 += factor * w

    new_bbox = [x1,y1,x2,y2]
    for i in range(len(new_bbox)):
        new_bbox[i] = int(np.round(new_bbox[i]))
    if new_bbox[0] < 0:
        new_bbox[0] = 0
    if new_bbox[1] < 0:
        new_bbox[1] = 0
    if new_bbox[2] >= limit[0]:
        new_bbox[2] = limit[0] - 1
    if new_bbox[3] >= limit[1]:
        new_bbox[3] = limit[1] - 1

    return tuple(new_bbox)


def find_daugs(track, track_id):
    """Return list of daughters according to certain parent track ID.

    Args:
        track (pandas.DataFrame): tracked object table.
        track_id (int): track ID.
    """
    rt = list(np.unique(track.loc[track['parentTrackId'] == track_id, 'trackId']))
    if not rt:
        return []
    else:
        to_rt = rt.copy()
        for trk in rt:
            to_rt.extend(find_daugs(track, trk))
        return to_rt
