"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import os
import yaml
import pandas as pd
import skimage.io as io
from ._utils import get_annotation


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str
        Path to input directory.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if not isinstance(path, str):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to input directory

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    cfgname = 'config.yaml'

    # handle sample data request
    if path == 'simple' or path == 'full':
        if path == 'full':
            cfgname = 'config_full.yaml'
        #  download from some online source
        #  store the sample data under home directory
        home = os.environ['HOME']
        root = os.path.join(home, '.amdtrk')
        if not os.path.isdir(root):
            os.makedirs(root)
        cwd = os.getcwd()
        os.chdir(root)
        flg = False
        if '_sample_data' not in os.listdir('.'):
            flg = True
        else:
            if len(os.listdir('./_sample_data')) < 6:
                # broken sample data
                flg = True
                os.system('rm -r ./_sample_data')

        if flg:
            os.system('wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1RV2lNRM6Yb5aOb-Mu0kbqI1_1SeObQgI\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p\')&id=1AMa-JWSa8u2oGI-kCLt79uPkI5plhF7f\" -O test.tar.gz && rm -rf /tmp/cookies.txt')
            os.system('tar -zxvf test.tar.gz -C .')
            os.system('rm test.tar.gz')
        path = os.path.join(root, '_sample_data')


    # look for config
    try:
        with open(os.path.join(path, cfgname), 'r') as f:
            cfg = yaml.safe_load(f.read())
    except:
        raise FileNotFoundError('Missing config file.')
    
    intensity_path, mask_path, track_path = '', '', ''
    for fname in os.listdir(path):
        sfx = fname.split('.')[0].split('_')[-1]
        if sfx == cfg['intensity_suffix']:
            intensity_path = os.path.join(path, fname)
        if sfx == cfg['mask_suffix']:
            mask_path = os.path.join(path, fname)
        if sfx == cfg['track_suffix']:
            track_path = os.path.join(path, fname)
    
    if intensity_path == '' or mask_path == '' or track_path == '':
        raise ValueError('Missing input file, check if filenames match the config.')
        
    stateCol = cfg['stateCol']
    track = pd.read_csv(track_path)

    # if no state column specified, will use a dummy column.
    if stateCol is None:
        hasState = False
        phaseVis = False
        stateColName = 'state'
        track['state'] = '0'
        states = ('0',)
    else:
        hasState = True
        phaseVis = True
        stateColName = stateCol
        statePool = track[stateCol]
        states = tuple(np.unique(statePool[~pd.isnull(statePool)]).astype('str'))
    states = list(states)

    def check_input_track(trk, hasState, stateColName):
        track = trk.copy()
        cols = list(track.columns)
        if ('lineageId' not in cols) ^ ('parentTrackId' not in cols):
            raise ValueError('The lineage and parent track ID should exist together!')
        if 'lineageId' not in cols:
            track['lineageId'] = track['trackId']
            track['parentTrackId'] = 0
        if 'name' not in cols:
            track = get_annotation(track, hasState, stateColName)
        return track
    
    track = check_input_track(track, hasState, stateColName)
    track = track.sort_values(by=['trackId','frame'])
    mask = io.imread(mask_path)

    rt = []
    if intensity_path is not None:
        comp = io.imread(intensity_path)
        if len(comp.shape) > 3:
            for i in range(comp.shape):
                rt.append((comp[:, :, :, i], {'name':'intensity' + str(i)}, 'image'))
        else:
            rt.append((comp, {'name':'intensity0'}, 'image'))

    rt.append((mask, {'name':'segm','metadata':{'frame_base': cfg['frame_base'], 'stateCol': stateCol, 
                        'stateColName': stateColName, 'track_path': track_path, 'phaseVis': phaseVis,
                        'mask_path': mask_path, 'states':states, 'hasState':hasState}}, 'labels'))
    track_data = track.loc[:][['trackId', 'frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
    label_data = track.loc[:][['frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
    label_data = label_data.to_numpy()
    track_data = track_data.to_numpy() # track layer only allow ndarray! pass track info to widget.
    text_config = {'text':'name', 'size': 8, 'color': 'yellow'}
    rt.append((track_data, {'name':'tracks', 'metadata':{'ori_data':track}}, 'tracks'))
    rt.append((label_data, {'name':'name', 'size':0,
                            'features':{'name':track.loc[:]['name'].to_numpy()}, 
                            'text':text_config}, 'points'))
    return rt
