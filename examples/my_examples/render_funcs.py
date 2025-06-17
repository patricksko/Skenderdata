import blenderproc as bproc
import os
import numpy as np
from typing import List, Tuple



def bounding_box_2d_from_vertices(object: bproc.types.MeshObject, K: np.ndarray, camTworld: np.ndarray) -> Tuple[List[float], float, np.ndarray]:
  '''
  Compute the 2D bounding from an object's vertices
  returns A tuple containing:
    [xmin, ymin, xmax, ymax] in pixels
    the proportion of visible vertices (that are not behind the camera, i.e., negative Z)
    The 2D points, in normalized units in camera space, in ViSP/OpenCV frame
  '''
  worldTobj = homogeneous_no_scaling(object)
  obj = object.blender_obj
  verts = np.ones(len(obj.data.vertices) * 3)
  obj.data.vertices.foreach_get("co", verts)
  points = verts.reshape(-1, 3)
  # verts = [v.co for v in obj.data.vertices]
  camTobj = camTworld @ worldTobj
  points_cam = camTobj @ np.concatenate((points, np.ones((len(points), 1))), axis=-1).T
  points_cam = convert_points_to_visp_frame((points_cam[:3] / points_cam[3, None]).T)

  visible_points = points_cam[points_cam[:, 2] > 0]
  visible_points_m_2d = visible_points[:, :2] / visible_points[:, 2, None]
  visible_points_px_2d = K @ np.concatenate((visible_points_m_2d, np.ones((len(visible_points_m_2d), 1))), axis=-1).T
  visible_points_px_2d = visible_points_px_2d.T[:, :2] / visible_points_px_2d.T[:, 2, None]

  mins = np.min(visible_points_px_2d, axis=0)
  assert len(mins) == 2
  maxes = np.max(visible_points_px_2d, axis=0)

  return [mins[0], mins[1], maxes[0], maxes[1]], len(visible_points) / len(points_cam), visible_points_m_2d

def homogeneous_inverse(aTb):
  '''
  Inverse of a 4x4 homogeneous transformation matrix
  '''
  bTa = aTb.copy()
  bTa[:3, :3] = bTa[:3, :3].T
  bTa[:3, 3] = -bTa[:3, :3] @ bTa[:3, 3]
  return bTa

def homogeneous_no_scaling(object: bproc.types.MeshObject, frame=None):
  '''
  Get the homogeneous transformation of an object, but without potential scaling
  object.local2world() may contain scaling factors.
  '''
  localTworld = np.eye(4)
  localTworld[:3, :3] = object.get_rotation_mat(frame)
  localTworld[:3, 3] = object.get_location(frame)
  return localTworld

def convert_points_to_visp_frame(points: np.ndarray):
  '''
  Convert a set of points to ViSP coordinate frame
  '''
  points = points.copy()
  points[:, 1:3] = -points[:, 1:3]
  return points