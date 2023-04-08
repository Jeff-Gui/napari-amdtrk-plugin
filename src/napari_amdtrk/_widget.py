"""
This module is a QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets
"""
from typing import TYPE_CHECKING
import warnings
from magicgui import magicgui
from magicgui.widgets import RadioButtons, Container
from qtpy.QtWidgets import QWidget
from ._utils import get_current_time, get_layer_id_by_name, find_daugs, align_table_and_mask, get_annotation
import numpy as np
import skimage.io as io
import skimage.measure as measure
import pandas as pd
import trackpy

if TYPE_CHECKING:
    import napari
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class AmdTrkWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer : 'napari.viewer.Viewer'):

        super().__init__()
        self.viewer = viewer
        self.segm_id = get_layer_id_by_name(self.viewer, 'segm')
        self.track_id = get_layer_id_by_name(self.viewer, 'tracks')
        self.name_id = get_layer_id_by_name(self.viewer, 'name')
        # meta: {'frame_base': int, 'stateCol': int, 'stateColName': str, 'track_path': str, 'mask_path': str}
        meta = self.viewer.layers[self.segm_id].metadata
        self.frame_base = meta['frame_base']
        self.stateCol = meta['stateCol']
        self.stateColName = meta['stateColName']
        self.track_path = meta['track_path']
        self.mask_path = meta['mask_path']
        phaseVis = meta['phaseVis']
        states = meta['states']
        self.hasState = meta['hasState']
        self.states = meta['states']

        self.track = self.viewer.layers[self.track_id].metadata['ori_data']
        self.saved = self.track.copy()
        self.mask = self.viewer.layers[self.segm_id].data.copy()
        self.track_count = int(np.max(self.track['trackId']))
        self.DILATE_FACTOR = int((self.viewer.layers[self.segm_id].data.shape[1] + 
                                  self.viewer.layers[self.segm_id].data.shape[2]) / 2 / 240)

        self.high = 255 if np.max(self.viewer.layers[self.segm_id].data) < 255 else 65536
        self.select = {}  # register selected obj (key: frame-label, value: (mask,contour))
        self.last_reg_id = 0
        self.label_unassigned = -1


        #================== Widget definitions =======================

        @magicgui(labels=False,
          auto_call=True,
          result_widget=True,
          rev={
              "widget_type": "PushButton",
              "text": "Revert",
          })
        def revert(rev):
            self.clear_selection()
            msg = self.revert()
            self.refresh()
            return msg

        @magicgui(labels=False,
                auto_call=True,
                result_widget=True,
                sv={
                    "widget_type": "PushButton",
                    "text": "Save",
                })
        def save(sv):
            self.clear_selection()
            msg = self.save()
            self.refresh()
            return msg
        
        @magicgui(labels=True, result_widget=True)
        def retrack(distance: int, frame_gap: int):
            self.clear_selection()
            msg = self.retrack(distance=distance, frame_gap=frame_gap)
            self.refresh()
            print('refreshed')
            return msg

        @magicgui(labels=True, result_widget=True)
        def create_or_replace(track_A: int, track_B: int=0, frame: int=0):
            self.clear_selection()
            self.check_assign([track_A, track_B] if track_B !=0 else [track_A])
            if track_B < 1:
                track_B = None
            msg = self.create_or_replace(track_A, frame, track_B)
            self.refresh()
            return msg

        @magicgui(labels=True, result_widget=True)
        def swap(track_A: int, track_B: int=0, frame: int=0):
            self.clear_selection()
            self.check_assign([track_A, track_B])
            if track_B < 1:
                raise ValueError('Must specify two tracks to swap!')
            msg = self.swap(track_A, frame, track_B)
            self.refresh()
            return msg

        @magicgui(labels=True, result_widget=True)
        def delete(track: int, frame: int=0):
            self.clear_selection()
            if frame < 1:
                frame = None
            msg = self.delete_track(track, frame)
            self.refresh()
            return msg

        @magicgui(labels=True, result_widget=True)
        def create_par(mother: int, daughter: int):
            self.clear_selection()
            self.check_assign([mother, daughter])
            msg = self.create_parent(mother, daughter)
            self.refresh()
            return msg

        @magicgui(labels=True, result_widget=True)
        def delete_par(daughter: int):
            self.clear_selection()
            self.check_assign([daughter])
            msg = self.del_parent(daughter)
            self.refresh()
            return msg
        
        @magicgui(labels=True, result_widget=True, IDs = {'widget_type':'TextEdit'})
        def keep_tracks(IDs:str = ''):
            self.clear_selection()
            msg = self.run_keep_tracks(list(map(lambda x:int(x), IDs.replace(' ', '').split(','))))
            self.refresh()
            return msg
        
        @magicgui(labels=True, result_widget=True)
        def copy_obj(ID: int, fromFrame: int, toFrame: int):
            self.clear_selection()
            msg = self.run_copy_obj(ID, fromFrame, toFrame)
            self.refresh()
            return msg

        @magicgui(labels=True,
                result_widget=True,
                state={
                    'widget_type': 'ComboBox',
                    'choices': states
                })
        def register_obj(object_ID: int, frame: int, track: int, state=states[0]):
            self.clear_selection()
            msg = self.register_obj(object_ID, frame, track, state)
            self.refresh()
            return msg

        @magicgui(labels=True,
                result_widget=True,
                mode={
                    'widget_type': 'RadioButtons',
                    'orientation': 'vertical',
                    'choices': [(' to next transition', 1), (' single frame', 2), (' to end frame', 3)]
                },
                phase={
                    'widget_type': 'ComboBox',
                    'choices': states
                })
        def phase(track: int, frame_start: int, frame_end: int, phase=states[0], mode=1):
            self.clear_selection()
            self.check_assign([track])
            mode_rev = {1: 'to_next', 2: 'single', 3: 'range'}
            if mode == 3:
                msg = self.correct_cls(track, frame_start, phase, mode_rev[mode], frame_end)
            else:
                msg = self.correct_cls(track, frame_start, phase, mode_rev[mode])
            self.refresh()
            return msg
        

        #================== Widget interactions =======================

        btnChoice = [(" Replace track A with B or \n Create track from certain frame", 1),
                                    (" Delete track", 2), (" Link mother - daughter", 3),
                                    (" Unlink mother - daughter", 4), (" Register object", 5),
                                    (" Swap track A with B", 6), (" Keep selected tracks", 7),
                                    (" Copy an object to another frame",8),
                                    (" Commit mask and re-track (TrackPy)", 9)]
        if phaseVis:
            btnChoice.append((" Edit state", 10))
        btns = RadioButtons(name='',
                            choices=btnChoice,
                            orientation='vertical',
                            label='',
                            )
        btns.value = 1
        @btns.changed.connect
        def _toggle_visibility(value: str):
            # helps to avoid a flicker
            for x in [create_or_replace, delete, phase, create_par, delete_par, 
                      register_obj, swap, keep_tracks, copy_obj, retrack]:
                x.visible = False
            create_or_replace.visible = value == 1
            delete.visible = value == 2
            phase.visible = value == 10
            create_par.visible = value == 3
            delete_par.visible = value == 4
            register_obj.visible = value == 5
            swap.visible = value == 6
            keep_tracks.visible = value == 7
            copy_obj.visible = value == 8
            retrack.visible = value == 9
            
        widget_map = {1:create_or_replace, 2:delete, 9:phase, 3:create_par, 4:delete_par, 
                      5:register_obj, 6:swap, 7:keep_tracks, 8:copy_obj, 9:retrack}
        if self.hasState:
            widget_map[10] = phase

        container_opt = Container(widgets=[btns, create_or_replace, delete, phase, create_par, 
                                           delete_par, register_obj, swap, keep_tracks, copy_obj, retrack],
                                layout='vertical',
                                labels=False)

        container_opt.margins = (0, 0, 0, 0)
        container_but = Container(widgets=[revert, save],
                                layout='horizontal',
                                labels=False)
        container_but.margins = (0, 0, 0, 0)
        container_ext = Container(widgets=[container_opt, container_but],
                                layout='vertical')
        container_ext.margins = (5, 5, 5, 5)
        # container.show(run=True)

        def reset_widget():
            
            nonlocal create_or_replace, delete, phase, create_par, delete_par, register_obj, keep_tracks, retrack
            
            # Default value of sub-widgets
            swap.update({'track_B':0, 'track_A':0, 'frame':0})
            create_or_replace.update({'track_B':0, 'track_A':0, 'frame':0})
            create_par.update({'daughter':0, 'mother':0})
            delete.update({'track':0, 'frame':0})
            phase.update({'track':0, 'frame_start':0})
            delete_par.update({'daughter':0})
            register_obj.update({'object_ID':0, 'frame':0, 'track':0, 'state': self.states[0]})
            keep_tracks.update({'IDs':''})
            copy_obj.update({'ID':0, 'fromFrame':0, 'toFrame':1})
            retrack.update({'distance':0, 'frame_gap':0})
        self.reset_widget = reset_widget

        self.viewer.add_shapes(name='[selection]', edge_width=2*self.DILATE_FACTOR, edge_color='coral', face_color=[0,0,0,0], ndim=3)
        
        labels = self.viewer.layers[self.segm_id]
        #sels = self.viewer.layers['[selection]']
        #trkly = self.viewer.layers[self.track_id]
        #namely = self.viewer.layers[self.name_id]
        
        #@sels.mouse_drag_callbacks.append
        @labels.mouse_drag_callbacks.append
        #@trkly.mouse_drag_callbacks.append
        #@namely.mouse_drag_callbacks.append
        def click_drag(layer, event):
            
            nonlocal create_or_replace, delete, phase, create_par, delete_par, register_obj, copy_obj, retrack
            # nonlocal labels
            # layer = labels
            if event.type == 'mouse_press' and layer.mode == 'pan_zoom':
                pos = event.position
                # label position, only work for txy (t+2D) data!
                pos = np.round(pos).astype('int')
                sle = layer.data[pos[0], :, :].copy()
                try:
                    _ = sle[pos[1], pos[2]]
                except: # if outside the image region
                    return
                if sle[pos[1], pos[2]] != 0:
                    lbl = sle[pos[1], pos[2]]
                    trk_id = self.track[(self.track['frame'] == pos[0]) & (self.track['continuous_label'] == lbl)]
                    if trk_id.shape[0]:
                        trk_id = trk_id['trackId'].iloc[0]
                    else:
                        trk_id = self.last_reg_id
                    if not self.hasState:
                        state = '0'
                    else:
                        state = self.track[(self.track['frame'] == pos[0]) & (self.track['continuous_label'] == lbl)]
                        if state.shape[0]:
                            state = state[self.stateColName].iloc[0]
                        else:
                            state = self.states[0]
                    ky = str(pos[0]) + '-' + str(lbl)
                    flg = False
                    if ky not in self.select.keys():  # select object
    
                        # update widget default values according to selections
                        #  involves two selection
                        slk = list(self.select.keys())
                        if len(slk) > 0:
                            create_or_replace.update({'track_B':trk_id})
                            swap.update({'track_B':trk_id, 'frame':pos[0]})
                            create_par.update({'daughter':trk_id})
                        else:
                            swap.update({'frame':pos[0], 'track_A':trk_id})
                            create_or_replace.update({'frame':pos[0], 'track_A':trk_id})
                            create_par.update({'mother':trk_id})
                        #  only one selection
                        delete.update({'track':trk_id, 'frame':pos[0]})
                        self.label_unassigned = lbl
                        phase.update({'track':trk_id, 'frame_start':pos[0]})
                        delete_par.update({'daughter':trk_id})
                        register_obj.update({'object_ID':lbl, 'frame':pos[0], 'track':trk_id, 'state': state})
                        copy_obj.update({'ID':lbl, 'fromFrame':pos[0], 'toFrame': pos[0] + 1})

                        # find the bounding box
                        msk = sle == lbl
                        minx, miny, maxx, maxy = measure.regionprops(measure.label(msk))[0].bbox
                        objBox = np.array([[pos[0], minx, miny], [pos[0], maxx, miny],
                                           [pos[0], maxx, maxy], [pos[0], minx, maxy]])
                        idx = len(viewer.layers['[selection]'].data)
                        new = viewer.layers['[selection]'].data.copy() + [objBox]
                        self.select[ky] = (objBox, idx)

                    else:  # delete object highlight (deselect)
                        
                        _, idx = self.select[ky]
                        new = viewer.layers['[selection]'].data.copy()
                        del new[idx]

                        # update widget default values according to selections
                        #  involves two selection
                        slk = list(self.select.keys())
                        if len(slk) > 1:
                            if ky == slk[0]:
                                # if the first selected object is deselected, directly back to the ground state
                                create_or_replace.update({'track_B':0, 'track_A':0, 'frame':0})
                                swap.update({'track_B':0, 'track_A':0, 'frame':0})
                                create_par.update({'daughter':0, 'mother':0})
                                flg = True
                            else:
                                # info = slk[-1].split('-')
                                # last_trk_id = track[track['frame'] == int(info[0]) & track['continuous_label'] == int(info[1])]['trackId'].iloc[0]
                                swap.update({'track_B':0})
                                create_or_replace.update({'track_B':0})
                                create_par.update({'daughter':0})
                        else:
                            swap.update({'frame':0, 'track_A':0})
                            create_or_replace.update({'frame':0, 'track_A':0})
                            create_par.update({'mother':0})
                        #  only one selection
                        delete.update({'track':0, 'frame':0})
                        self.label_unassigned = -1
                        phase.update({'track':0, 'frame_start':0})
                        delete_par.update({'daughter':0})
                        register_obj.update({'object_ID':0, 'frame':0, 'track':0, 'state': self.states[0]})
                        copy_obj.update({'ID':0, 'fromFrame':0, 'toFrame':1})

                        if not flg:
                            del self.select[ky]

                    if not flg:
                        self.viewer.layers['[selection]'].data = new.copy()
                    else:
                        self.clear_selection()
                else:
                    # update maximum label of the frame to title
                    vstatus = self.viewer.status
                    if isinstance(vstatus, dict):
                        vstatus = vstatus['coordinates']
                    fme = int(vstatus.split('[')[1].lstrip().split(' ')[0])
                    mxLabel = self.get_mx(fme)
                    self.viewer.title = 'AmdTrk | Max label of frame: ' + str(fme) + ' is ** ' + str(mxLabel) + ' **'
                    self.viewer.layers[self.segm_id].selected_label = mxLabel + 1
                    self.clear_selection()

        @self.viewer.bind_key('Enter')
        def _resolve_key(self):
            # run widget from keyboard
            nonlocal btns, widget_map
            wig = widget_map[btns.value]
            wig()
            return

        @self.viewer.bind_key('Up', overwrite=True)
        def _toggle_up(self):
            nonlocal btns, phaseVis
            if btns.value == 1:
                btns.value = 10 if phaseVis else 9
            else:
                    btns.value -= 1
            return
        
        @self.viewer.bind_key('Down', overwrite=True)
        def _toggle_down(self):
            nonlocal btns, phaseVis
            if btns.value == 10:
                btns.value = 1
            else:
                if not phaseVis and btns.value == 9:
                    btns.value = 1
                else:
                    btns.value += 1
            return

        # Update the layer dropdown menu when the layer list changes
        self.viewer.layers.events.changed.connect(container_ext.reset_choices)
        # Add plugin to the napari viewer
        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(container_ext)
        self.viewer.window.add_dock_widget(container_ext, area='left')
        return

    def clear_selection(self):
        if len(self.select.keys()) > 0:
            self.select = {}
            self.viewer.layers['[selection]'].data = []
            self.reset_widget()
        return
    
    def check_assign(self, IDs_to_check):
        for i in IDs_to_check:
            if i == 0:
                raise ValueError('Must first assign track ID to the object!')
        return


    #================== Widget functions =======================

    def create_or_replace(self, old_id, frame, new_id=None):
        """Create a new track ID or replace with some track ID
        after certain frame. If the old track has daughters, new track ID will be the parent.

        Args:
            old_id (int): old track ID.
            frame (int): frame to begin with new ID.
            new_id (int): new track ID, only required when replacing track identity.
        """
        if old_id not in self.track['trackId'].values:
            raise ValueError('Selected track is not in the table.')
        if frame not in list(self.track[self.track['trackId'] == old_id]['frame']):
            raise ValueError('Selected frame is not in the original track.')
        relabel = False
        if new_id not in self.track['trackId'].values:
            # raise ValueError('Selected new ID not in the table.')
            relabel = True

        dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == old_id, 'trackId']))
        for dd in dir_daugs:
            self.del_parent(dd)

        if new_id is None:
            self.track_count += 1
            new = self.track_count
            new_lin = new
            new_par = 0
        else:
            new = new_id
            if relabel:
                new_lin = self.track[self.track['trackId'] == old_id]['lineageId'].values[0]
                if new_lin == old_id: # if this track is a root track, or is not involved in mitosis
                    new_lin = new
                new_par = self.track[self.track['trackId'] == old_id]['parentTrackId'].values[0]
            else:
                old_frame = list(self.track[self.track['trackId'] == new_id]['frame'])
                new_frame = list(self.track.loc[(self.track['trackId'] == old_id) &
                                                (self.track['frame'] >= frame), 'frame'])
                if len(old_frame + new_frame) != len(set(old_frame + new_frame)):
                    raise ValueError('Selected new ID track overlaps with old one.')
                new_lin = self.track[self.track['trackId'] == new_id]['lineageId'].values[0]
                new_par = self.track[self.track['trackId'] == new_id]['parentTrackId'].values[0]
        
        self.track.loc[(self.track['trackId'] == old_id) & (self.track['frame'] >= frame), 'trackId'] = new
        self.track.loc[self.track['trackId'] == new, 'lineageId'] = new_lin
        self.track.loc[self.track['trackId'] == new, 'parentTrackId'] = new_par
        
        # daughters of the new track, change lineage
        if not relabel:
            daugs = find_daugs(self.track, new)
            if daugs:
                self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = new_lin
        for dd in dir_daugs:
            if dd != new:
                self.create_parent(new, dd)
        
        msg = 'Track ' + str(old_id) + ' from frame ' + str(frame + self.frame_base) + \
              ' <- Track ' + str(new) + '.'
        print(msg)
        return msg

    def swap(self, track_A, frame, track_B):
        """Swap track A with track B after certain frame. If the old track has daughters, new track ID will be the parent.

        Args:
            track_A (int): track ID A.
            frame (int): frame to begin with new ID.
            track_B (int): track ID B.
        """
        if track_A not in self.track['trackId'].values:
            raise ValueError('Selected track is not in the table.')
        if track_B not in self.track['trackId'].values:
            raise ValueError('Selected track is not in the table.')
        if frame not in list(self.track[self.track['trackId'] == track_A]['frame']):
            raise ValueError('Selected frame is not in the original track.')

        dir_daugs_A = list(np.unique(self.track.loc[self.track['parentTrackId'] == track_A, 'trackId']))
        for dd in dir_daugs_A:
            self.del_parent(dd)
        dir_daugs_B = list(np.unique(self.track.loc[self.track['parentTrackId'] == track_B, 'trackId']))
        for dd in dir_daugs_B:
            self.del_parent(dd)

        self.track['trackId'] = self.track['trackId'].astype('str')
        track_A, track_B = str(track_A), str(track_B)
        new_A = track_B + '-' + track_A
        new_B = track_A + '-' + track_B
        new_A_lin = self.track[self.track['trackId'] == track_B]['lineageId'].values[0]
        new_A_par = self.track[self.track['trackId'] == track_B]['parentTrackId'].values[0]
        new_B_lin = self.track[self.track['trackId'] == track_A]['lineageId'].values[0]
        new_B_par = self.track[self.track['trackId'] == track_A]['parentTrackId'].values[0]
        
        self.track.loc[(self.track['trackId'] == track_A) & (self.track['frame'] >= frame), 'trackId'] = new_A
        self.track.loc[self.track['trackId'] == new_A, 'lineageId'] = new_A_lin
        self.track.loc[self.track['trackId'] == new_A, 'parentTrackId'] = new_A_par
        self.track.loc[(self.track['trackId'] == track_B) & (self.track['frame'] >= frame), 'trackId'] = new_B
        self.track.loc[self.track['trackId'] == new_B, 'lineageId'] = new_B_lin
        self.track.loc[self.track['trackId'] == new_B, 'parentTrackId'] = new_B_par

        self.track.loc[(self.track['trackId'] == new_A) & (self.track['frame'] >= frame), 'trackId'] = track_B
        self.track.loc[(self.track['trackId'] == new_B) & (self.track['frame'] >= frame), 'trackId'] = track_A
        
        self.track['trackId'] = self.track['trackId'].astype('int')
        track_A, track_B = int(track_A), int(track_B)
        # daughters of the new track, change lineage
        daugs = find_daugs(self.track, track_B)
        if daugs:
            self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = new_A_lin
        for dd in dir_daugs_A:
            if dd != track_B:
                self.create_parent(track_B, dd)
        daugs = find_daugs(self.track, track_A)
        if daugs:
            self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = new_B_lin
        for dd in dir_daugs_B:
            if dd != track_A:
                self.create_parent(track_A, dd)
        
        msg = 'Track ' + str(track_A) + ' from frame ' + str(frame + self.frame_base) + \
              ' <- swapped with Track ' + str(track_B) + '.'
        print(msg)
        return msg

    def create_parent(self, par, daug):
        """Create parent-daughter relationship.

        Args:
            par (int): parent track ID.
            daug (int): daughter track ID.
        """
        if par not in self.track['trackId'].values:
            raise ValueError('Selected parent is not in the table.')
        if daug not in self.track['trackId'].values:
            raise ValueError('Selected daughter is not in the table.')

        ori_par = self.track[self.track['trackId'] == daug]['parentTrackId'].iloc[0]
        if ori_par != 0:
            raise ValueError('One daughter cannot have more than one parent, disassociate ' + str(ori_par) + '-'
                             + str(daug) + ' first.')

        par_lin = self.track[self.track['trackId'] == par]['lineageId'].iloc[0]
        # daughter itself
        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = par_lin
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = par
        # daughter of the daughter
        daugs_of_daug = find_daugs(self.track, daug)
        if daugs_of_daug:
            self.track.loc[self.track['trackId'].isin(daugs_of_daug), 'lineageId'] = par_lin

        msg =  'Track ' + str(par) + ' linked with ' + str(daug) + '.'
        print(msg)
        return msg

    def del_parent(self, daug):
        """Remove parent-daughter relationship, for a daughter.

        Args:
            daug (int): daughter track ID.
        """
        if daug not in self.track['trackId'].values:
            raise ValueError('Selected daughter is not in the table.')
        if self.track[self.track['trackId'] == daug]['parentTrackId'].iloc[0] == 0:
            raise ValueError('Selected daughter does not have a parent.')

        # daughter itself
        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = daug
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = 0
        # daughters of the daughter, change lineage
        daugs = find_daugs(self.track, daug)
        if daugs:
            self.track.loc[self.track['trackId'].isin(daugs), 'lineageId'] = daug

        msg = 'Track ' + str(daug) + ' unlinked from its mother.'
        print(msg)
        return msg

    def correct_cls(self, trk_id, frame, cls, mode='to_next', end_frame=None):
        """Correct state classification.

        Args:
            trk_id (int): track ID to correct.
            frame (int): frame to correct or begin with correction.
            cls (str): new state classification ID to assign.
            mode (str): either 'to_next', 'single', or 'range'
            end_frame (int): optional, in 'range' mode, stop correction at this frame.
        """
        if trk_id not in self.track['trackId'].values:
            raise ValueError('Selected track is not in the table.')
        if cls not in self.states:
            raise ValueError('Input state ID not registered.')

        clss = list(self.track[self.track['trackId'] == trk_id][self.stateColName])
        frames = list(self.track[self.track['trackId'] == trk_id]['frame'])
        if frame not in frames:
            raise ValueError('Selected frame is not in the original track.')
        fm_id = frames.index(frame)
        idx = self.track[self.track['trackId'] == trk_id].index
        if mode == 'single':
            rg = [fm_id]
        elif mode == 'range':
            if end_frame not in frames:
                raise ValueError('Selected end frame is not in the original track.')
            rg = [i for i in range(fm_id, frames.index(end_frame + 1))]
        elif mode == 'to_next':
            cur_cls = clss[fm_id]
            j = fm_id + 1
            while j < len(clss):
                if clss[j] == cur_cls:
                    j += 1
                else:
                    break
            rg = [i for i in range(fm_id, j)]
        else:
            raise ValueError('Mode can only be single, to_next or range, not ' + mode)

        for r in rg:
            self.track.loc[idx[r], self.stateColName] = cls
        msg = 'Track ' + str(trk_id) + ' state <- ' + str(cls) + ' from ' + \
              str(frames[rg[0]] + self.frame_base) + ' to ' + str(frames[rg[-1]] + self.frame_base) + '.'
        print(msg)
        return msg

    def delete_track(self, trk_id, frame=None):
        """Delete entire track. If frame supplied, only delete object at specified frame.

        Args:
            trk_id (int): track ID.
            frame (int): time frame.
        """
        del_unreg_sel = False
        if trk_id not in self.track['trackId'].values:
            if trk_id == 0:
                del_unreg_sel = True
            else:
                raise ValueError('Selected track is not in the table.')

        mask = self.viewer.layers[self.segm_id].data
        if not del_unreg_sel:
            del_trk = self.track[self.track['trackId'] == trk_id]
        if frame is None and not del_unreg_sel:

            if trk_id != 0:
                # For all direct daughter of the track to delete, first remove association
                dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == trk_id, 'trackId']))
                for dd in dir_daugs:
                    self.del_parent(dd)
            else:
                warnings.warn('Deleting all unassigned objects in all frames')

            # Delete entire track
            for i in range(del_trk.shape[0]):
                fme = del_trk['frame'].iloc[i]
                lb = del_trk['continuous_label'].iloc[i]
                msk_slice = mask[fme, :, :]
                msk_slice[msk_slice == lb] = 0
                mask[fme, :, :] = msk_slice
            self.track = self.track.drop(index=del_trk.index)
            msg = 'Deleted track ' + str(trk_id) + '.'
        else:
            if trk_id != 0:
                del_trk = del_trk[del_trk['frame'] == frame]
                lb = del_trk['continuous_label'].iloc[0]
            else:
                lb = self.label_unassigned
            msk_slice = mask[frame, :, :]
            msk_slice[msk_slice == lb] = 0
            mask[frame, :, :] = msk_slice
            if trk_id != 0:
                self.track = self.track.drop(index=self.track[(self.track['trackId'] == trk_id) &
                                                              (self.track['frame'] == frame)].index)
                msg = 'Deleted track ' + str(trk_id) + ' at frame ' + str(frame) + '.'
            elif not del_unreg_sel:
                self.track = self.track.drop(index=self.track[(self.track['trackId'] == trk_id) &
                                                              (self.track['frame'] == frame) &
                                                              (self.track['continuous_label'] == lb)].index)
                msg = 'Deleted unassigned object ' + str(lb) + ' at frame ' + str(frame) + '.'
            else:
                msg = 'Deleted an unregistered object.'
        self.viewer.layers[self.segm_id].data = mask
        print(msg)
        return msg

    def run_keep_tracks(self, trk_ids):
        """Only keep tracks specified in the input list.
        """
        mask = self.viewer.layers[self.segm_id].data

        for trk_id in trk_ids:
            if trk_id in self.track['trackId']:  # no warning if input track is not in the dataset.
                # For all direct daughters of the track to keep, if not in input list, dissociate.
                dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == trk_id, 'trackId']))
                for dd in dir_daugs:
                    if dd not in trk_ids:
                        self.del_parent(dd)
                # For parent, if not in input list, dissociate
                par = list(self.track.loc[self.track['trackId'] == trk_id, 'parentTrackId'])[0]
                if par != 0 and par not in trk_ids:
                    self.del_parent(trk_id)
        
        self.track = self.track[self.track['trackId'].isin(trk_ids)]
        for frame in range(mask.shape[0]):
            lb = list(self.track[self.track['frame'] == frame]['continuous_label'])
            msk_slice = mask[frame, :, :]
            new_mask = np.zeros(shape=msk_slice.shape, dtype=mask.dtype)
            for l in lb:
                new_mask[msk_slice == l] = l
            mask[frame, :, :] = new_mask
    
        self.viewer.layers[self.segm_id].data = mask

        msg = 'Tracks kept: ' + ','.join(list(map(lambda x:str(x), trk_ids))) + '.'
        print(msg)
        return msg

    def save(self, mask_flag=True):
        """Save current table.
        """
        mask = self.viewer.layers[self.segm_id].data
        track = self.track.copy()
        if mask_flag:
            mask, track = align_table_and_mask(track, mask, align_morph=False, 
                                               phase_col=self.stateColName, phase_default=self.states[0])      # warning: align_morph=False
            if int(np.max(mask)) <= 255:
                io.imsave(self.mask_path, mask.astype('uint8'))
            else:
                io.imsave(self.mask_path, mask)
        self.mask = mask.copy()
        self.getAnn()
        track = track.sort_values(by=['trackId', 'frame'])
        track.to_csv(self.track_path, index=None)
        self.saved = track.copy()
        self.track = track.copy()
        msg = 'Saved: ' + get_current_time() + '.'
        return msg

    def revert(self):
        """Revert to last saved version.
        """
        self.viewer.layers[self.segm_id].data = self.mask.copy()
        self.track = self.saved.copy()
        msg = 'Reverted: ' + get_current_time() + '.'
        return msg
    
    def retrack(self, distance, frame_gap):
        trk = self.track.copy()
        mask = self.viewer.layers[self.segm_id].data
        mask, trk = align_table_and_mask(trk, mask, align_morph=False)   
        trk['index'] = trk.index
        t = trackpy.link(trk[['frame', 'Center_of_the_object_0', 'Center_of_the_object_1', 'index']], 
                         search_range=distance, memory=frame_gap, adaptive_stop=0.4*distance, 
                         pos_columns=['Center_of_the_object_0', 'Center_of_the_object_1'])
        cols = list(t.columns)
        cols[len(cols)-1] = 'trackId'
        t.columns = cols
        trk.loc[t['index'], 'trackId'] = t['trackId'] + 1   # trackpy output start from ID=0
        
        trk['lineageId'] = trk['trackId']
        trk['parentTrackId'] = 0            # TODO resolve previously associated mitosis
        del trk['index']
        trk = trk.sort_values(by=['trackId', 'frame'])
        trk.index = [_ for _ in range(trk.shape[0])]

        self.mask = mask.copy()
        self.track = trk.copy()
        msg = 'Re-tracked.'
        return msg

    def getAnn(self):
        """Add an annotation column to tracked object table
        The annotation format is track ID - (parentTrackId, optional) - stateColName
        """
        track = self.track.copy()
        if self.hasState:
            track.loc[pd.isnull(track[self.stateColName]), self.stateColName] = self.states[0]
        track = get_annotation(track, self.hasState, self.stateColName)
        self.track = track
        return

    def edit_div(self, par, daugs, new_frame):
        """Change division time of parent and daughter to a new time location
        TODO: check and implement as a widget function

        Args:
            par (int): parent track ID
            daugs (list): daughter tracks IDs
            new_frame (int):
        """
        if par not in self.track['trackId'].values:
            raise ValueError('Selected parent track is not in the table.')
        for d in daugs:
            if d not in self.track['trackId'].values:
                raise ValueError('Selected daughter track is not in the table.')
            if self.track[self.track['trackId'] == d]['parentTrackId'].iloc[0] != par:
                raise ValueError('Selected daughter track does not corresponding to the input parent.')

        new_frame -= 1
        sub_par = self.track[self.track['trackId'] == par]
        time_daugs = []
        sub_daugs = pd.DataFrame()
        for i in daugs:
            sub_daugs = pd.concat([sub_daugs, self.track[self.track['trackId'] == i]], ignore_index=True)
        time_daugs.extend(list(sub_daugs['frame']))
        if new_frame not in list(sub_par['frame']) and new_frame not in time_daugs:
            raise ValueError('Selected new time frame not in either parent or daughter track.')

        if new_frame not in list(sub_par['frame']):
            # push division later
            edit = sub_daugs[sub_daugs['frame'] <= new_frame]
            if len(np.unique(edit['trackId'])) > 1:
                raise ValueError('Multiple daughters at selected new division, should only have one')

            # get and assign edit index
            par_id = sub_par['trackId'].iloc[0]
            par_lin = sub_par['lineageId'].iloc[0]
            par_par = sub_par['parentTrackId'].iloc[0]
            self.track.loc[edit.index, 'trackId'] = par_id
            self.track.loc[edit.index, 'lineageId'] = par_lin
            self.track.loc[edit.index, 'parentTrackId'] = par_par
        else:
            new_frame += 1
            # draw division earlier
            edit = sub_par[sub_par['frame'] >= new_frame]
            # pick a daughter that appears earlier and assign tracks to that daughter
            f_min = np.argmin(sub_daugs['frame'])
            if len(np.unique(sub_daugs[sub_daugs['frame'] == f_min]['trackId'])) > 1:
                raise ValueError('Multiple daughters exist at frame of mitosis, should only be one. '
                                 'Or break mitotic track first.')
            trk = list(sub_daugs['trackId'])[int(f_min)]
            sel_daugs = sub_daugs[sub_daugs['trackId'] == trk]

            # get and assign edit index
            daug_id = sel_daugs['trackId'].iloc[0]
            daug_par = sel_daugs['parentTrackId'].iloc[0]
            daug_lin = sel_daugs['lineageId'].iloc[0]
            self.track.loc[edit.index, 'trackId'] = daug_id
            self.track.loc[edit.index, 'parentTrackId'] = daug_par
            self.track.loc[edit.index, 'lineageId'] = daug_lin

        return

    def register_obj(self, obj_id, frame, trk_id, cls):
        """Register a new object that has been drawn on the mask

        Args:
            obj_id (int): Object ID of the drawn mask.
            frame (int): Frame location.
            trk_id (int): Track ID.
            cls (str): State id.
        """
        mask = self.viewer.layers[self.segm_id].data
        msk_slice = mask[frame, :, :]
        trk_slice = self.track[self.track['frame'] == frame]
        untracked = False if int(trk_id) > 0 else True          # register as an untracked object
        if obj_id in list(trk_slice['continuous_label']):
            if list(trk_slice[trk_slice['continuous_label'] == obj_id]['trackId'])[0] == 0:
                # assign information to untracked object
                idx = list(trk_slice[trk_slice['continuous_label'] == obj_id]['trackId'].index)[0]
                self.track.loc[idx, 'trackId'] = trk_id
                self.track.loc[idx, 'lineageId'] = trk_id
                if self.hasState:
                    self.track.loc[idx, self.stateColName] = cls
                self.track = self.track.sort_values(by=['trackId', 'frame'])
                msg = 'Assign obj: track ' + str(trk_id) + '; frame ' + str(frame) + '; state ' + cls + '.'
                self.last_reg_id = trk_id
                return msg
            else:
                raise ValueError('Object label has been used, draw with a bigger label. Current max label: ' +
                                  str(np.max(trk_slice['continuous_label'])))
        if trk_id in list(trk_slice['trackId']) and trk_id != 0:
            raise ValueError('Track ID already exists in the selected frame.')
        if cls not in self.states:
            raise ValueError('Given state ID not registered.')
        if obj_id not in msk_slice:
            raise ValueError('Object ID is not in the given frame of mask. Draw again. Current max label: ' +
                             str(np.max(trk_slice['continuous_label'])))

        new_row = {'frame': frame, 'trackId': trk_id, 'continuous_label': obj_id,
                   # below fields are not essential for the input and should have been init by default value already
                   self.stateColName: cls}
        if trk_id in list(self.track['trackId']):
            # pld track
            old_trk = self.track[self.track['trackId'] == trk_id]
            old_lin = old_trk['lineageId'].iloc[0]
            old_par = old_trk['parentTrackId'].iloc[0]
            new_row['lineageId'] = old_lin
            new_row['parentTrackId'] = old_par
            if old_par != 0:
                nm = '-'.join([str(trk_id), str(old_par), cls])
            else:
                nm = '-'.join([str(trk_id), cls])
        elif untracked:
            new_row['lineageId'] = 0
            new_row['trackId'] = 0
            new_row['parentTrackId'] = 0
            nm = 'unassigned'
        else:
            # New track
            self.track_count = int(np.max((self.track_count, trk_id)))
            new_row['lineageId'] = trk_id
            new_row['parentTrackId'] = 0
            nm = '-'.join([str(trk_id), cls])
        new_row['name'] = nm

        # Register measurements of the object morphology.
        misc = np.zeros(msk_slice.shape, dtype='uint8')
        misc[msk_slice == obj_id] = 1
        p = measure.regionprops(label_image=misc)[0]
        y, x = p.centroid
        new_row['Center_of_the_object_0'] = x
        new_row['Center_of_the_object_1'] = y
        # For extra fields
        for i in set(list(self.track.columns)) - set(list(new_row.keys())):
            new_row[i] = np.nan
        
        self.track = pd.concat([self.track, pd.DataFrame.from_dict([new_row])[self.track.columns]], ignore_index=True)
        self.track = self.track.sort_values(by=['trackId', 'frame'])
        if trk_id != 0:
            msg = 'New obj: track ' + str(trk_id) + '; frame ' + str(frame) + '; state ' + cls + '.'
        else:
            msg = 'New unassigned obj: ' + 'frame ' + str(frame) + '; state ' + cls + '.'
        self.last_reg_id = trk_id
        return msg

    def run_copy_obj(self, ID, fromFrame, toFrame):
        # copy object (labeled as ID) from frame A to frame B, will overlap on existing objects on B.
        if fromFrame == toFrame:
            raise ValueError('Cannot copy object on the same frame.')
        row = self.track[(self.track['frame'] == fromFrame) & (self.track['continuous_label'] == ID)]
        mask = self.viewer.layers[self.segm_id].data
        if row.shape[0] != 1:
            # If unassigned object found in fromFrame, register it first.
            if ID in np.unique(mask[fromFrame,:,:]):
                self.register_obj(obj_id=ID, frame=fromFrame, trk_id=0, cls=self.states[0])
            else:
                raise ValueError('ID not found in fromFrame.')
        
        tar_mx = self.get_mx(toFrame)
        row.loc[row.index, 'continuous_label'] = tar_mx + 1
        row.loc[row.index, 'frame'] = toFrame
        new_track = pd.concat([self.track, row], ignore_index=True)
        new_track = new_track.sort_values(by=['trackId','frame'])
        new_track.index = [_ for _ in range(new_track.shape[0])]
        self.track = new_track
        toMask = mask[toFrame,:,:].copy()
        fromMask = mask[fromFrame,:,:]
        toMask[fromMask==ID] = tar_mx + 1
        mask[toFrame,:,:] = toMask
        self.viewer.layers[self.segm_id].data = mask
        msg = ''
        return msg

    def get_mx(self, frame):
        mask = self.viewer.layers[self.segm_id].data
        return int(np.max(mask[frame,:,:]))

    def refresh(self):
        self.getAnn()
        track_data = self.track.loc[:][['trackId', 'frame', 'Center_of_the_object_1', 'Center_of_the_object_0']].copy()
        track_data = track_data[track_data['trackId']>0] # unassigned tracks have ID=0, not allowed for napari to plot.
        track_data = track_data.to_numpy().astype('float')
        self.viewer.layers[self.track_id].data = track_data
        label_data = self.track.loc[:][['frame', 'Center_of_the_object_1', 'Center_of_the_object_0']]
        label_data = label_data.to_numpy()
        # self.layers[nm_idx].features.clear()
        self.viewer.layers[self.name_id].data = label_data
        self.viewer.layers[self.name_id].features['name'] = self.track.loc[:]['name'].to_numpy()
        self.viewer.layers[self.name_id].refresh_text()
        self.clear_selection()
        return
