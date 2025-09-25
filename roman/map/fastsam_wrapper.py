#########################################
# 
# fastsam_wrapper.py
#
# A Python wrapper for sending RGBD images to FastSAM and using segmentation 
# masks to create object observations.
# 
# Authors: Jouko Kinnari, Mason Peterson, Lucas Jia, Annika Thomas
# 
# Dec. 21, 2024
#
#########################################

import clip
import copy
import cv2 as cv
import math
import numpy as np
import open3d as o3d
import random
import torch
from torch.amp import autocast
from yolov7_package import Yolov7Detector
from PIL import Image
from fastsam import FastSAMPrompt
from fastsam import FastSAM
import logging
from robotdatapy.camera import CameraParams
from robotdatapy.transform import transform
from roman.map.observation import Observation
from roman.params.fastsam_params import FastSAMParams
from roman.utils import expandvars_recursive
import rerun as rr
from ..scene_graph.logger import logger 
from scipy.ndimage import uniform_filter
from scipy.spatial.transform import Rotation as R


from typeguard import typechecked

# Try to make FastSAM and YOLO Deterministic
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@typechecked
def mask_bounding_box(mask: np.ndarray) -> tuple:
    """ 
    Calculate the bounding box around a mask.
    Originally by Annika Thomas.
    """

    # Find the indices of the True values
    true_indices = np.argwhere(mask)

    if len(true_indices) == 0:
        # No True values found, return None or an appropriate response
        return None

    # Calculate the mean of the indices
    mean_coords = np.mean(true_indices, axis=0)

    # Calculate the width and height based on the min and max indices in each dimension
    min_row, min_col = np.min(true_indices, axis=0)
    max_row, max_col = np.max(true_indices, axis=0)
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Define a bounding box around the mean coordinates with the calculated width and height
    min_row = int(max(mean_coords[0] - height // 2, 0))
    max_row = int(min(mean_coords[0] + height // 2, mask.shape[0] - 1))
    min_col = int(max(mean_coords[1] - width // 2, 0))
    max_col = int(min(mean_coords[1] + width // 2, mask.shape[1] - 1))

    return (min_col, min_row, max_col, max_row,)

class FastSAMWrapper():

    def __init__(self, weights: str, conf: float =.5, iou: float =.9, imgsz: tuple[int, int] =(1024, 1024),
        device: str ='cuda', mask_downsample_factor: int = 1, rotate_img=None) -> None:
        """Wrapper for running FastSAM on images (especially RGB/depth pairs)

        Args:
            weights (str): Path to FastSAM weights.
            conf (float, optional): FastSAM confidence threshold. Defaults to .5.
            iou (float, optional): FastSAM IOU threshold. Defaults to .9.
            imgsz (tuple, optional): Image size to feed into FastSAM. Defaults to (1024, 1024).
            device (str, optional): 'cuda' or 'cpu. Defaults to 'cuda'.
            mask_downsample_factor (int, optional): For creating smaller data observations. 
                Defaults to 1.
            rotate_img (_type_, optional): 'CW', 'CCW', or '180' for rotating image before 
                feeding into FastSAM. Defaults to None.
        """
        # parameters
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.mask_downsample_factor = mask_downsample_factor
        self.rotate_img = rotate_img

        # For calculating Scene Flow
        self._last_pc = None
        self._last_pose = None
        self.frame_tick = 0
        self.send_to_rerun = False
        
        # member variables
        self.observations = []
        self.model = FastSAM(weights)

        # setup default filtering
        self.setup_filtering()

        assert self.device == 'cuda' or self.device == 'cpu', "Device should be 'cuda' or 'cpu'."
        assert self.rotate_img is None or self.rotate_img == 'CW' or self.rotate_img == 'CCW' \
            or self.rotate_img == '180', "Invalid rotate_img option."
            
    @classmethod
    def from_params(cls, params: FastSAMParams, depth_cam_params: CameraParams):
        fastsam = cls(
            weights=expandvars_recursive(params.weights_path),
            imgsz=params.imgsz,
            device=params.device,
            mask_downsample_factor=params.mask_downsample_factor,
            rotate_img=params.rotate_img
        )
        fastsam.setup_rgbd_params(
            depth_cam_params=depth_cam_params, 
            max_depth=params.max_depth,
            depth_scale=params.depth_scale,
            depth_data_type=params.depth_data_type,
            voxel_size=params.voxel_size,
            erosion_size=params.erosion_size,
            plane_filter_params=params.plane_filter_params
        )

        img_area = depth_cam_params.width * depth_cam_params.height
        fastsam.setup_filtering(
            ignore_labels=params.ignore_labels,
            use_keep_labels=params.use_keep_labels,
            keep_labels=params.keep_labels,
            keep_labels_option=params.keep_labels_option,
            yolo_weights=expandvars_recursive(params.yolo_weights_path),
            yolo_det_img_size=params.yolo_imgsz,
            allow_tblr_edges=[True, True, True, True],
            area_bounds=[img_area / (params.min_mask_len_div**2), img_area / (params.max_mask_len_div**2)],
            clip_embedding=params.clip,
            triangle_ignore_masks=params.triangle_ignore_masks
        )

        return fastsam
            
    def setup_filtering(self, ignore_labels: list = [], use_keep_labels=False, keep_labels: list = [], keep_labels_option='intersect',          
        yolo_weights=None, yolo_det_img_size=None, area_bounds=np.array([0, np.inf]), allow_tblr_edges = [True, True, True, True],
        keep_mask_minimal_intersection=0.3, clip_embedding=False, clip_model='ViT-L/14', triangle_ignore_masks=None) -> None:
        """
        Filtering setup function

        Args:
            ignore_labels (list, optional): List of yolo labels to ignore masks. Defaults to [].
            use_keep_labels (bool, optional): Use list of labels to only keep masks within keep mask. Defaults to False.
            keep_labels (list, optional): List of yolo labels to keep masks. Defaults to [].
            keep_labels_option (str, optional): 'intersect' or 'contain'. Defaults to 'intersect'.
            yolo_det_img_size (List[int], optional): Two-item list denoting yolo image size. Defaults to None.
            area_bounds (np.array, shape=(2,), optional): Two element array indicating min and max number of pixels. Defaults to np.array([0, np.inf]).
            allow_tblr_edges (list, optional): Allow masks touching top, bottom, left, and right edge. Defaults to [True, True, True, True].
            keep_mask_minimal_intersection (float, optional): Minimal intersection of mask within keep mask to be kept. Defaults to 0.3.
        """
        assert not use_keep_labels or keep_labels_option == 'intersect' or keep_labels_option == 'contain', "Keep labels option should be one of: intersect, contain"
        self.ignore_labels = ignore_labels
        self.use_keep_labels = use_keep_labels
        self.keep_labels = keep_labels
        self.keep_labels_option=keep_labels_option
        if len(ignore_labels) > 0 or use_keep_labels:
            if yolo_det_img_size is None:
                yolo_det_img_size=self.imgsz
            self.yolov7_det = Yolov7Detector(traced=False, img_size=yolo_det_img_size, weights=yolo_weights)
        
        self.area_bounds = area_bounds
        self.allow_tblr_edges= allow_tblr_edges
        self.keep_mask_minimal_intersection = keep_mask_minimal_intersection
        self.run_yolo = len(ignore_labels) > 0 or use_keep_labels
        
        self.clip_embedding = clip_embedding
        if clip_embedding:
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
            self.clip_model.eval()
        if triangle_ignore_masks is not None:
            self.constant_ignore_mask = np.zeros((self.depth_cam_params.height, self.depth_cam_params.width), dtype=np.uint8)
            for triangle in triangle_ignore_masks:
                assert len(triangle) == 3, "Triangle must have 3 points."
                for pt in triangle:
                    assert len(pt) == 2, "Each point must have 2 coordinates."
                    assert all([isinstance(x, int) for x in pt]), "Coordinates must be integers."
                cv.fillPoly(self.constant_ignore_mask, [np.array(triangle)], 1)
            self.constant_ignore_mask = self.apply_rotation(self.constant_ignore_mask)
        else:
            self.constant_ignore_mask = None
            
    def setup_rgbd_params(self, depth_cam_params: CameraParams, max_depth: float, depth_data_type: str, depth_scale: float = 1e3, voxel_size: float = 0.05, 
        within_depth_frac: float = 0.5, pcd_stride: int = 4, erosion_size = 0, plane_filter_params = None) -> None:
        """Setup params for processing RGB-D depth measurements

        Args:
            depth_cam_params (CameraParams): parameters of depth camera
            max_depth (float): maximum depth to be included in point cloud
            depth_scale (float, optional): scale of depth image. Defaults to 1e3.
            voxel_size (float, optional): Voxel size when downsampling point cloud. Defaults to 0.05.
            within_depth_frac(float, optional): Fraction of points that must be within max_depth. Defaults to 0.5.
            pcd_stride (int, optional): Stride for downsampling point cloud. Defaults to 4.
            plane_filter_params (List[float], optional): If an object's oriented bounding box's extent from max to min is > > <, mask is rejected. Defaults to None.
        """
        self.depth_cam_params = depth_cam_params
        self.max_depth = max_depth
        self.within_depth_frac = within_depth_frac
        self.depth_scale = depth_scale
        self.depth_data_type = getattr(np, depth_data_type)
        self.depth_cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(depth_cam_params.width),
            height=int(depth_cam_params.height),
            fx=depth_cam_params.fx,
            fy=depth_cam_params.fy,
            cx=depth_cam_params.cx,
            cy=depth_cam_params.cy,
        )
        self.voxel_size = voxel_size
        self.pcd_stride = pcd_stride
        if erosion_size > 0:
            # see: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
            erosion_shape = cv.MORPH_ELLIPSE
            self.erosion_element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                (erosion_size, erosion_size))
        else:
            self.erosion_element = None
        self.plane_filter_params = plane_filter_params

    @typechecked
    def run(self, t: float, pose: np.ndarray, img: np.ndarray, img_depth: np.ndarray[float] = None) -> list[Observation]:
        """ Takes an image and returns filtered FastSAM masks as Observations. """

        self.observations: list[Observation] = []
        
        # rotate image
        img_orig = img
        img = self.apply_rotation(img)

        if self.run_yolo:
            ignore_mask, keep_mask = self._create_mask(img)
        else:
            ignore_mask = None
            keep_mask = None

        if self.constant_ignore_mask is not None:
            ignore_mask = np.bitwise_or(ignore_mask, self.constant_ignore_mask) \
                if ignore_mask is not None else self.constant_ignore_mask  
        
        # Run FastSAM
        masks: np.ndarray = self._process_img(img, ignore_mask=ignore_mask, keep_mask=keep_mask)
        
        # Remove masks corresponding to dynamic objects
        # masks = self.remove_dynamic_object_masks(masks, img_depth, pose)

        # ================== Generate Observations ==================
        for i, mask in enumerate(masks):
            mask = self.apply_rotation(mask, unrotate=True)

            # ============= Extract point cloud of object from RGBD =============
            ptcld = None
            if img_depth is not None:

                # Set depth to zero everywhere except detected object
                depth_obj = copy.deepcopy(img_depth)
                if self.erosion_element is not None:
                    eroded_mask = cv.erode(mask, self.erosion_element)
                    depth_obj[eroded_mask==0] = 0
                else:
                    depth_obj[mask==0] = 0

                # Extract point cloud without truncation to heuristically check if enough of the object
                # is within the max depth
                pcd_test = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d.geometry.Image(np.ascontiguousarray(depth_obj).astype(self.depth_data_type)),
                    self.depth_cam_intrinsics,
                    depth_scale=self.depth_scale,
                    stride=self.pcd_stride,
                    project_valid_depth_only=True
                )
                ptcld_test = np.asarray(pcd_test.points)
                pre_truncate_len = len(ptcld_test)
                ptcld_test = ptcld_test[ptcld_test[:,2] < self.max_depth]
                # require some fraction of the points to be within the max depth
                if len(ptcld_test) < self.within_depth_frac*pre_truncate_len:
                    continue
                
                # Remove points past max depth, downsample, and remove non-finite points
                pcd_test.remove_non_finite_points()

                # Remove this, as I want to downsample related to size with Scene Graph
                # pcd_test = pcd_test.voxel_down_sample(voxel_size=self.voxel_size)

                if not pcd_test.is_empty():
                    ptcld = np.asarray(pcd_test.points)
                    ptcld = ptcld[ptcld[:,2] <= self.max_depth]
                if ptcld is None:
                    continue
                
                if self.plane_filter_params is not None:
                    # Create oriented bounding box
                    try:
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                                o3d.utility.Vector3dVector(ptcld))
                        extent = np.sort(obb.extent)[::-1] # in descending order
                        if  extent[0] > self.plane_filter_params[0] and \
                            extent[1] > self.plane_filter_params[1] and \
                            extent[2] < self.plane_filter_params[2]:
                                continue
                    except:
                        continue
  
            # Generate downsampled mask
            mask_downsampled = np.array(cv.resize(
                mask,
                (mask.shape[1]//self.mask_downsample_factor, mask.shape[0]//self.mask_downsample_factor), 
                interpolation=cv.INTER_NEAREST
            )).astype('uint8')

            # Save the observation
            self.observations.append(Observation(t, pose, mask, mask_downsampled, ptcld))

        # ===== Generate CLIP embeddings for remaining observations in a single batch =====
        if self.clip_embedding and len(self.observations) > 0:

            # Get processed mask images
            processed_imgs_list = []
            for i, obs in enumerate(self.observations):

                # Calculate bounding box around mask
                bbox = mask_bounding_box(obs.mask.astype('uint8'))
                if bbox is None: raise RuntimeError("Bounding Box is None")

                # Get processed image of this mask
                min_col, min_row, max_col, max_row = bbox
                img_bbox = self.apply_rotation(img_orig[min_row:max_row, min_col:max_col])
                img_bbox = cv.cvtColor(img_bbox, cv.COLOR_BGR2RGB)
                processed_imgs_list.append(self.clip_preprocess(Image.fromarray(img_bbox, mode='RGB')))
            processed_imgs = torch.stack(processed_imgs_list).to(self.device)
            
            # Calculate CLIP embeddings
            clip_embeddings: np.ndarray | None = None
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    clip_embeddings = self.clip_model.encode_image(processed_imgs).cpu().detach().numpy()  

            # Assign embeddings to observations
            for i, obs in enumerate(self.observations):
                obs.clip_embedding = clip_embeddings[i]

        return self.observations
    
    @typechecked
    def remove_dynamic_object_masks(self, masks: np.ndarray, img_depth: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """ Uses Scene Flow to detect masks of dynamic object and remove them. """

        # Skip computation if we detect no objects
        if len(masks) == 0: return masks

        # Skip Scene Flow calculation on first frame
        if self._last_pc is not None and self._last_pose is not None:

            # Transform last point cloud into current frame
            T_curr_wrt_last = np.linalg.inv(self._last_pose) @ pose
            last_pc_wrt_curr = transform(np.linalg.inv(T_curr_wrt_last), self._last_pc, axis=0)

            # Project transformed points into our current image frame
            x, y, z = last_pc_wrt_curr[:,0], last_pc_wrt_curr[:,1], last_pc_wrt_curr[:,2]
            u = (self.depth_cam_params.fx * x / z + self.depth_cam_params.cx).astype(int)
            v = (self.depth_cam_params.fy * y / z + self.depth_cam_params.cy).astype(int)

            # Filter valid pixel locations
            H, W = img_depth.shape
            valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u_valid, v_valid = u[valid], v[valid]
            last_pc_wrt_curr = last_pc_wrt_curr[valid]

            # Convert depth at (u, v) into 3D using intrinsics
            depth_curr = img_depth[v_valid, u_valid]
            depth_curr = np.clip(depth_curr, 0, self.max_depth)
            x_curr = (u_valid - self.depth_cam_params.cx) * depth_curr / self.depth_cam_params.fx
            y_curr = (v_valid - self.depth_cam_params.cy) * depth_curr / self.depth_cam_params.fy
            z_curr = depth_curr
            pc_curr_matched = np.stack([x_curr, y_curr, z_curr], axis=1)

            # Calculate scene flow
            scene_flow = pc_curr_matched - last_pc_wrt_curr
            flow_mag = np.linalg.norm(scene_flow, axis=1)

            # Set the threshold, which is calculated as expected translation and rotation error based on estimated odometry,
            # and includes term to account for imprecision when aligning last points to current pixels
            trans_norm = np.linalg.norm(T_curr_wrt_last[:3,3])
            trans_error = 0.05 * trans_norm

            rotation_norm = np.linalg.norm(R.from_matrix(T_curr_wrt_last[:3,:3]).as_rotvec())
            rotation_error = 0.05 * rotation_norm
            chord_length = 2 * depth_curr * math.sin(rotation_error / 2)

            pixel_imprecision = 0.1 * depth_curr

            threshold = trans_error + chord_length + pixel_imprecision

            # Calculate dynamic object mask (must have flow above threshold and be within depth range)
            is_dynamic = (flow_mag > threshold).astype(bool) & (depth_curr < self.max_depth).astype(bool)
            u_dyn = u_valid[is_dynamic]
            v_dyn = v_valid[is_dynamic]
            
            # Send data to Rerun for visualization
            if self.send_to_rerun:
                rr.set_time("fastsam_frame_tick", sequence=self.frame_tick)
                rr.log("/fastsam/points/curr", rr.Points3D(positions=pc_curr_matched[is_dynamic]))
                rr.log("/fastsam/points/last", rr.Points3D(positions=last_pc_wrt_curr[is_dynamic]))
                
                dynamic_mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
                kernel = np.ones((11, 11), np.uint8)
                dynamic_mask_vis[v_dyn, u_dyn] = np.array([255, 255, 255], dtype=np.uint8)
                dynamic_mask_vis = cv.dilate(dynamic_mask_vis, kernel)
                rr.log("/fastsam/camera/depth", rr.DepthImage(img_depth))
                rr.log("/fastsam/camera/mask", rr.Image(dynamic_mask_vis))
                
                # Visualize threshold and flow_mag (sparse image)
                threshold_image = np.zeros((H, W), dtype=np.float32)
                flow_mag_image = np.zeros((H, W), dtype=np.float32)
                threshold_image[v_valid, u_valid] = threshold
                flow_mag_image[v_valid, u_valid] = flow_mag
                threshold_image = cv.dilate(threshold_image, kernel)
                flow_mag_image = cv.dilate(flow_mag_image, kernel)
                rr.log("/fastsam/camera/thresh", rr.DepthImage(threshold_image, meter=1.0))
                rr.log("/fastsam/camera/flow_mag", rr.DepthImage(flow_mag_image, meter=1.0))

                self.frame_tick += 1

            # ============= Remove FastSAM Detections overlapping with dynamic mask =============
            to_keep = []
            for i, mask in enumerate(masks):

                # Get dynamic point indices that lie within the mask
                mask_flat = mask.astype(bool)
                valid_in_mask = mask_flat[v_valid, u_valid]
                dyn_in_mask = mask_flat[v_dyn, u_dyn]

                num_valid_in_mask = np.count_nonzero(valid_in_mask)
                if num_valid_in_mask > 0:
                    overlap_ratio = np.count_nonzero(dyn_in_mask) / num_valid_in_mask
                else: overlap_ratio = 0.0 # We saw no part of this object last frame

                if overlap_ratio < 0.5:
                    to_keep.append(i)
                else:
                    logger.info(f"[red]Discard[/red]: Mask {i} detected as a dynamic object.")
                
            masks = masks[to_keep,...]

        # Generate point cloud at reduced resolution for next frame
        masked_depth = np.clip(img_depth, 0, self.max_depth)
        point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(np.ascontiguousarray(masked_depth).astype(np.float32)), self.depth_cam_intrinsics,
            depth_scale=self.depth_scale, stride=10, project_valid_depth_only=True)
        point_cloud: np.ndarray = np.asarray(point_cloud.points)

        # Save current point cloud and pose for next frame calculating scene flow
        self._last_pc = point_cloud
        self._last_pose = pose

        return masks
    
    def apply_rotation(self, img, unrotate=False):
        if self.rotate_img is None:
            result = img
        elif self.rotate_img == 'CW':
            result = cv.rotate(img, cv.ROTATE_90_CLOCKWISE 
                               if not unrotate else cv.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate_img == 'CCW':
            result = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE 
                               if not unrotate else cv.ROTATE_90_CLOCKWISE)
        elif self.rotate_img == '180':
            result = cv.rotate(img, cv.ROTATE_180)
        else:
            raise Exception("Invalid rotate_img option.")
        return result
        
    def _create_mask(self, img):
        
        if len(img.shape) == 2: # image is mono
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        classes, boxes, scores = self.yolov7_det.detect(img)
        ignore_boxes = []
        keep_boxes = []
        for i, cl in enumerate(classes[0]):
            if self.yolov7_det.names[cl] in self.ignore_labels:
                ignore_boxes.append(boxes[0][i])

            if self.yolov7_det.names[cl] in self.keep_labels:
                keep_boxes.append(boxes[0][i])

        ignore_mask = np.zeros(img.shape[:2]).astype(np.int8)
        for box in ignore_boxes:
            x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
            box_before_truncation = np.array([x0, y0, x1, y1])
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, ignore_mask.shape[1])
            y1 = min(y1, ignore_mask.shape[0])

            try:
                ignore_mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
            except:
                print("Ignore box: ", box_before_truncation)
                print("Ignore box after truncating: ", x0, y0, x1, y1)
                print("Ignore mask shape: ", ignore_mask.shape)
                raise Exception("Invalid ignore box.") 
    

        if self.use_keep_labels:
            keep_mask = np.zeros(img.shape[:2]).astype(np.int8)
            for box in keep_boxes:
                x0, y0, x1, y1 = np.array(box).astype(np.int64).reshape(-1).tolist()
                x0 = max(x0, 0)
                y0 = max(y0, 0)
                x1 = min(x1, keep_mask.shape[1])
                y1 = min(y1, keep_mask.shape[0])
                keep_mask[y0:y1,x0:x1] = np.ones((y1-y0, x1-x0)).astype(np.int8)
        else:
            keep_mask = None

        return ignore_mask, keep_mask

    def _delete_edge_masks(self, segmask):
        [numMasks, h, w] = segmask.shape
        contains_edge = np.zeros(numMasks).astype(np.bool_)
        for i in range(numMasks):
            mask = segmask[i,:,:]
            edge_width = 5
            # TODO: should be a parameter
            contains_edge[i] = (np.sum(mask[:,:edge_width]) > 0 and not self.allow_tblr_edges[2]) or (np.sum(mask[:,-edge_width:]) > 0 and not self.allow_tblr_edges[3]) or \
                            (np.sum(mask[:edge_width,:]) > 0 and not self.allow_tblr_edges[0]) or (np.sum(mask[-edge_width:, :]) > 0 and not self.allow_tblr_edges[1])
        return np.delete(segmask, contains_edge, axis=0)

    def _process_img(self, image_bgr, ignore_mask=None, keep_mask=None) -> np.ndarray:
        """Process FastSAM on image, returns segment masks and center points from results

        Args:
            image_bgr ((h,w,3) np.array): color image
            fastSamModel (FastSAM): FastSAM object
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
            plot (bool, optional): Plots (slow) for visualization. Defaults to False.
            ignore_edges (bool, optional): Filters out edge-touching segments. Defaults to False.

        Returns:
            segmask ((n,h,w) np.array): n segmented masks (binary mask over image)
            blob_means ((n, 2) list): pixel means of segmasks
            blob_covs ((n, (2, 2) np.array) list): list of covariances (ellipses describing segmasks)
            (fig, ax) (Matplotlib fig, ax): fig and ax with visualization
        """

        # OpenCV uses BGR images, but FastSAM and Matplotlib require an RGB image, so convert.
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

        # Run FastSAM
        with torch.no_grad():
             with autocast(device_type='cuda', dtype=torch.float16):
                everything_results = self.model(image, 
                                                retina_masks=True, 
                                                device=self.device, 
                                                imgsz=self.imgsz, 
                                                conf=self.conf, 
                                                iou=self.iou)
                prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
                segmask = prompt_process.everything_prompt()

        # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
        if (len(segmask) > 0):
            segmask = segmask.cpu().numpy()
        else:
            segmask = None

        if (segmask is not None):
            # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
            # of the detected segment
            [numMasks, h, w] = segmask.shape

            # filter out edge-touching segments
            # could do with multi-dimensional summing faster instead of looping over masks
            if not np.all(self.allow_tblr_edges):
                segmask = self._delete_edge_masks(segmask)
                [numMasks, h, w] = segmask.shape

            to_delete = []
            for maskId in range(numMasks):
                # Extract the single binary mask for this mask id
                mask_this_id = segmask[maskId,:,:]

                # filter out ignore mask
                if ignore_mask is not None and np.any(np.bitwise_and(mask_this_id.astype(np.int8), ignore_mask)):
                    to_delete.append(maskId)
                    continue

                # Only keep masks that are within keep_mask
                # if keep_mask is not None and not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask)):
                #     print("Delete maskID: ", maskId)
                #     to_delete.append(maskId)
                #     continue
                # if keep_mask is not None and self.keep_labels_option == 'intersect' and (not np.any(np.bitwise_and(mask_this_id.astype(np.int8), keep_mask))):
                if keep_mask is not None and self.keep_labels_option == 'intersect' and (np.bitwise_and(mask_this_id.astype(np.int8), keep_mask).sum() < self.keep_mask_minimal_intersection*mask_this_id.astype(np.int8).sum()):
                    to_delete.append(maskId)
                    continue

                if self.area_bounds is not None:
                    area = np.sum(mask_this_id)
                    if area < self.area_bounds[0] or area > self.area_bounds[1]:
                        to_delete.append(maskId)
                        continue

            segmask = np.delete(segmask, to_delete, axis=0)

        else: 
            h, w, _ = image_bgr.shape
            return np.zeros((0, h, w), dtype=bool)

        return segmask
    