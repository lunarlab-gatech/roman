###########################################################
#
# mapper.py
#
# ROMAN open-set segment mapper class
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import cv2 as cv
import open3d as o3d

from robotdatapy.data.img_data import CameraParams

from roman.object.segment import Segment
from roman.map.observation import Observation
from roman.map.global_nearest_neighbor import global_nearest_neighbor
from roman.map.map import ROMANMap
from roman.params.mapper_params import MapperParams
from roman.logger import logger


class Mapper():

    def __init__(self, params: MapperParams, camera_params: CameraParams):
        self.params = params
        self.camera_params = camera_params

        self.segment_nursery = []
        self.segments = []
        self.inactive_segments = []
        self.segment_graveyard = []
        self.id_counter = 0
        self.last_pose = None
        self.poses_flu_history = []
        self.times_history = []
        self._T_camera_flu = np.eye(4)

    def update(self, t: float, pose: np.array, observations: List[Observation]):

        # have T_WC, want T_WB
        # T_WB = T_WC @ T_CB
        self.poses_flu_history.append(pose @ self._T_camera_flu)
        self.times_history.append(t)
        
        # store last pose
        self.last_pose = pose.copy()

        # associate observations with segments
        # mask_similarity = lambda seg, obs: max(self.mask_similarity(seg, obs, projected=False), 
        #                                        self.mask_similarity(seg, obs, projected=True))

        for i, obs in enumerate(observations):
            logger.info(f"Observation {i} num of points: {obs.point_cloud.shape[0]}")

        #segments_to_associate = sorted(self.segments, key=lambda s: s.id) + sorted(self.segment_nursery, key=lambda s: s.id)
        associated_pairs = global_nearest_neighbor(
            self.segments + self.segment_nursery, observations, self.voxel_grid_similarity, self.params.min_iou
        )

        # Print associated pairs
        for pair in associated_pairs:
            logger.info(f"Association (seg, obs): ({(self.segments + self.segment_nursery)[pair[0]].id} {pair[1]})")

        # separate segments associated with nursery and normal segments
        pairs_existing = [[seg_idx, obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx < len(self.segments)]
        pairs_nursery = [[seg_idx - len(self.segments), obs_idx] for seg_idx, obs_idx \
                                in associated_pairs if seg_idx >= len(self.segments)]

        # update segments with associated observations
        for seg_idx, obs_idx in pairs_existing:
            self.segments[seg_idx].update(observations[obs_idx], integrate_points=True)
            # if self.segments[seg_idx].num_points == 0:
            #     self.segments.pop(seg_idx)
        for seg_idx, obs_idx in pairs_nursery:
            # forcing add does not try to reconstruct the segment
            self.segment_nursery[seg_idx].update(observations[obs_idx], integrate_points=True)
            # if self.segment_nursery[seg_idx].num_points == 0:
            #     self.segment_nursery.pop(seg_idx)

        # delete masks for segments that were not seen in this frame
        for seg in self.segments:
            if not np.allclose(t, seg.last_seen, rtol=0.0):
                seg.last_observation.mask = None

        # handle moving existing segments to inactive
        to_rm = [seg for seg in self.segments \
                    if t - seg.last_seen > self.params.max_t_no_sightings \
                        or seg.num_points == 0]
        for seg in to_rm:
            if seg.num_points == 0:
                logger.info(f"Deleting Segment {seg.id} due to too few points (during inactivation)")
                self.segments.remove(seg)
                continue
            try:
                logger.debug(f"Node Point cloud: {seg.points[0:3]}")
                logger.info(f"Moving Segment {seg.id} from normal to inactive")
                seg.final_cleanup(epsilon=self.params.segment_voxel_size*5.0)
                logger.debug(f"Node Point cloud: {seg.points[0:3]}")
                self.inactive_segments.append(seg)
                self.segments.remove(seg)
            except: # too few points to form clusters
                logger.info(f"Deleting Segment {seg.id} due to too few points to form clusters (during inactivation)")
                self.segments.remove(seg)
            
        # handle moving inactive segments to graveyard
        to_rm = [seg for seg in self.inactive_segments \
                    if t - seg.last_seen > self.params.segment_graveyard_time \
                        or np.linalg.norm(seg.last_observation.pose[:3,3] - pose[:3,3]) \
                            > self.params.segment_graveyard_dist]
        for seg in to_rm:
            logger.info(f"Downgrading Segment {seg.id} from inactive to graveyard")
            self.segment_graveyard.append(seg)
            self.inactive_segments.remove(seg)

        to_rm = [seg for seg in self.segment_nursery \
                    if t - seg.last_seen > self.params.max_t_no_sightings \
                        or seg.num_points == 0]
        for seg in to_rm:
            logger.info("Removing Segment {} from nursery due to no sightings or zero points".format(seg.id))
            self.segment_nursery.remove(seg)

        # handle moving segments from nursery to normal segments
        to_upgrade = [seg for seg in self.segment_nursery \
                        if seg.num_sightings >= self.params.min_sightings]
        for seg in to_upgrade:
            logger.info(f"Upgrading Segment {seg.id} from nursery to normal")
            self.segment_nursery.remove(seg)
            self.segments.append(seg)

        # add new segments
        associated_obs = [obs_idx for _, obs_idx in associated_pairs]
        new_observations: list[tuple[Observation, int]] = [(obs, idx) for idx, obs in enumerate(observations) \
                            if idx not in associated_obs]

        for obs, idx in new_observations:
            new_seg = Segment(obs, self.camera_params, self.id_counter, self.params.segment_voxel_size)
            if new_seg.num_points == 0: # guard from observations coming in with no points
                logger.info(f"Observation {idx} discarded as it has no points")
                continue
            logger.info(f"Observation {idx} turned into Segment {new_seg.id}")
            self.segment_nursery.append(new_seg)
            self.id_counter += 1

        # Print Ids of nodes in each category
        logger.debug(f"Segment nursery Ids: {[seg.id for seg in self.segment_nursery]}")
        logger.debug(f"Active segments Ids: {[seg.id for seg in self.segments]}")
        logger.debug(f"Inactive segments Ids: {[seg.id for seg in self.inactive_segments]}")
        logger.debug(f"Segment graveyard Ids: {[seg.id for seg in self.segment_graveyard]}")

        logger.debug(f"Starting merging process")
        self.merge(t)
    
        # Print Ids of nodes in each category
        logger.debug(f"Segment nursery Ids: {sorted([seg.id for seg in self.segment_nursery])}")
        logger.debug(f"Active segments Ids: {sorted([seg.id for seg in self.segments])}")
        logger.debug(f"Inactive segments Ids: {sorted([seg.id for seg in self.inactive_segments])}")
        logger.debug(f"Segment graveyard Ids: {sorted([seg.id for seg in self.segment_graveyard])}")

        # Print the current number of points in each segment
        for seg in self.segments:
            logger.debug(f"Segment {seg.id}: num_points={seg.num_points}")
        for seg in self.segment_nursery:
            logger.debug(f"Nursery Segment {seg.id}: num_points={seg.num_points}")
        for seg in self.inactive_segments:
            logger.debug(f"Inactive Segment {seg.id}: num_points={seg.num_points}")
        for seg in self.segment_graveyard:
            logger.debug(f"Graveyard Segment {seg.id}: num_points={seg.num_points}")
        return
    
    def voxel_grid_similarity(self, segment: Segment, observation: Observation, observation_id: int):
        """
        Compute the similarity between the voxel grids of a segment and an observation
        """
        voxel_size = self.params.iou_voxel_size
        logger.debug(f"Voxel Size: {voxel_size}")
        segment_voxel_grid = segment.get_voxel_grid(voxel_size)
        observation_voxel_grid = observation.get_voxel_grid(voxel_size)
        if segment.id == 2 and observation_id ==0:
            logger.debug(f"Seg: {segment_voxel_grid}")
            logger.debug(f"Obs: {observation_voxel_grid}")
        return segment_voxel_grid.iou(observation_voxel_grid)

    def mask_similarity(self, segment: Segment, observation: Observation, projected: bool = False):
        """
        Compute the similarity between the mask of a segment and an observation
        """
        if not projected or segment in self.segment_nursery:
            segment_propagated_mask = segment.last_observation.mask_downsampled
            # segment_propagated_mask = segment.propagated_last_mask(observation.time, observation.pose, downsample_factor=self.mask_downsample_factor)
            if segment_propagated_mask is None:
                iou = 0.0
            else:
                iou = Mapper.compute_iou(segment_propagated_mask, observation.mask_downsampled)

        # compute the similarity using the projected mask rather than last mask
        else:
            segment_mask = segment.reconstruct_mask(observation.pose, 
                            downsample_factor=self.params.mask_downsample_factor)
            iou = Mapper.compute_iou(segment_mask, observation.mask_downsampled)
        return iou
    
    @staticmethod
    def compute_iou(mask1, mask2):
        """Compute the intersection over union (IoU) of two masks.

        Args:
            mask1 (_type_): _description_
            mask2 (_type_): _description_
        """

        assert mask1.shape == mask2.shape
        logger.debug(f"Compute IoU for shape {mask1.shape}")
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if np.isclose(union, 0):
            return 0.0
        return float(intersection) / float(union)

    def remove_bad_segments(self, segments: List[Segment], min_volume: float=0.0, min_max_extent: float=0.0, plane_prune_params: List[float]=[np.inf, np.inf, 0.0]):
        """
        Remove segments that have small volumes or have no points

        Args:
            segments (List[Segment]): List of segments
            min_volume (float, optional): Minimum allowable segment volume. Defaults to 0.0.

        Returns:
            segments (List[Segment]): Filtered list of segments
        """
        to_delete = []
        # reason = []
        for seg in segments:
            if seg.id == 816:
                logger.info("DATA FOR SEG 816 in remove")
                logger.info(np.sort(seg.extent))
                logger.info(seg.num_points)
            try:
                extent = np.sort(seg.extent) # in ascending order
                if seg.num_points == 0:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has no points")
                elif seg.volume < min_volume:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has volume {seg.volume} < {min_volume}")
                elif extent[-1] < min_max_extent:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has max extent {np.max(seg.extent)} < {min_max_extent}"
                elif extent[2] > plane_prune_params[0] and extent[1] > plane_prune_params[1] and extent[0] < plane_prune_params[2]:
                    to_delete.append(seg)
                    # reason.append(f"Segment {seg.id} has extent {seg.extent} which is likely a plane")
            except: 
                to_delete.append(seg)
                # reason.append(f"Segment {seg.id} has an error in extent/volume computation")
        for seg in to_delete:
            # Print the id of the segment we're deleting
            logger.info(f"Deleting segment {seg.id} as it is a bad segment")
            segments.remove(seg)
            # for r in reason:
            #     print(r)
        return segments
    
    def merge(self, t):
        """
        Merge segments with high overlap
        """

        # Right now existing segments are merged with other existing segments or 
        # segments inthe graveyard. Heuristic for merging involves either projected IOU or 
        # 3D IOU. Should look into more.

        max_iter = 100
        n = 0
        edited = True

        self.inactive_segments = self.remove_bad_segments(
            self.inactive_segments, 
            min_max_extent=self.params.min_max_extent, 
            plane_prune_params=self.params.plane_prune_params
        )
        self.segments = self.remove_bad_segments(self.segments)

        # repeatedly try to merge until no further merges are possible
        while n < max_iter and edited:
            edited = False
            n += 1
            
            # Enable to mimic MeronomyGraph Disabled. Set to False to mimic original ROMAN baseline.
            if self.params.sort_segments_during_merge:
                segments_list = sorted(self.segments, key=lambda s: s.id)
                inactive_segments_list = sorted(self.segments, key=lambda s: s.id) + sorted(self.inactive_segments, key=lambda s: s.id)
            else:
                segments_list = self.segments
                inactive_segments_list = self.inactive_segments

            # TODO: This was changed temporarily. Will need to change back AND ensure Meronomy matches!
            for i, seg1 in enumerate(segments_list):
                for j, seg2 in enumerate(inactive_segments_list):
                    if i >= j:
                        continue

                    # if segments are very far away, don't worry about doing extra checking
                    if np.mean(seg1.points) - np.mean(seg2.points) > \
                        .5 * (np.max(seg1.extent) + np.max(seg2.extent)):
                        continue 

                    maks1 = seg1.reconstruct_mask(self.last_pose)
                    maks2 = seg2.reconstruct_mask(self.last_pose)
                    intersection2d = np.logical_and(maks1, maks2).sum()
                    union2d = np.logical_or(maks1, maks2).sum()
                    iou2d = intersection2d / union2d

                    if seg1.id == 301 and seg2.id == 291:
                        logger.debug(f"{seg1.id} Point cloud: {seg1.points[0:3]}")
                        logger.debug(f"{seg2.id} Point cloud: {seg2.points[0:3]}")
                        if t == 1665777947.8362014:
                            np.save(f"roman1_{n}.npy", seg1.points)
                            np.save(f"roman2_{n}.npy", seg2.points)

                        logger.debug(f"A Voxel Grid: {seg1.get_voxel_grid(self.params.iou_voxel_size)}")
                        logger.debug(f"B Voxel Grid: {seg2.get_voxel_grid(self.params.iou_voxel_size)}")

                    iou3d = seg1.get_voxel_grid(self.params.iou_voxel_size).iou(
                        seg2.get_voxel_grid(self.params.iou_voxel_size))
                    
                    if seg1.id == 301 and seg2.id == 291:
                        logger.debug(f"MERGING CHECK: Seg {seg1.id} and Seg {seg2.id} with 3D IOU {iou3d} and 2D IOU {iou2d}")

                    if iou3d > self.params.merge_objects_iou_3d or iou2d > self.params.merge_objects_iou_2d:
                        logger.info(f"Merging segments {seg1.id} and {seg2.id} with 3D IoU {iou3d:.2f} and 2D IoU {iou2d:.2f}")
                        seg1.update_from_segment(seg2)
                        seg1.id = min(seg1.id, seg2.id)
                        if seg1.num_points == 0:
                            self.segments.remove(seg1)
                        elif j < len(self.segments):
                            self.segments.remove(seg2)
                        else:
                            self.inactive_segments.remove(seg2)
                        edited = True
                        break
                if edited:
                    break
        return
            
    def make_pickle_compatible(self):
        """
        Make the Mapper object pickle compatible
        """
        for seg in self.segments + self.segment_nursery + self.inactive_segments + self.segment_graveyard:
            seg.reset_obb()
        return
    
    def get_segment_map(self) -> List[Segment]:
        """
        Get the segment map
        """
        segment_map = self.remove_bad_segments(
            self.segment_graveyard + self.inactive_segments + 
            self.segments)
        for seg in segment_map:
            seg.reset_obb()
        return segment_map
    
    def get_roman_map(self) -> ROMANMap:
        """
        Return the full ROMAN map.

        Returns:
            ROMANMap: Map of objects
        """
        segment_map = self.get_segment_map()
        return ROMANMap(
            segments=segment_map,
            trajectory=self.poses_flu_history,
            times=self.times_history,
            poses_are_flu=True
        )
    
    def set_T_camera_flu(self, T_camera_flu: np.array):
        """
        Set the transformation matrix from camera frame to forward-left-up frame
        """
        self._T_camera_flu = T_camera_flu
        return
    
    @property
    def T_camera_flu(self):
        return self._T_camera_flu
            