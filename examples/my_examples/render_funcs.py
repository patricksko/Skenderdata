import numpy as np
import cv2
from PIL import Image
import blenderproc as bproc


def sample_pose_func(obj: bproc.types.MeshObject):
    """Sample random pose for an object"""
    min_loc = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max_loc = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min_loc, max_loc))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def mask_to_polygons(mask):
    """Convert binary mask to YOLO polygon format"""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Minimum 3 points for a valid polygon
            polygon = contour.reshape(-1, 2).flatten().tolist()
            if len(polygon) >= 6:  # Minimum 3 points (6 coordinates)
                polygons.append(polygon)
    return polygons

def visualize_mask_and_polygons(rgb_img, mask, polygons, obj_name, save_path):
    """Create a visualization showing the original image, binary mask, and polygon overlay"""
    img_height, img_width = mask.shape
    
    # Create a 2x2 subplot visualization
    fig_height, fig_width = img_height * 2, img_width * 2
    visualization = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
    
    # Top-left: Original RGB image
    visualization[:img_height, :img_width] = rgb_img
    
    # Top-right: Binary mask (white on black)
    mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    visualization[:img_height, img_width:] = mask_rgb
    
    # Bottom-left: RGB image with mask overlay (semi-transparent red)
    overlay_img = rgb_img.copy()
    red_overlay = np.zeros_like(rgb_img)
    red_overlay[:, :, 0] = mask * 255  # Red channel
    overlay_img = cv2.addWeighted(overlay_img, 0.7, red_overlay, 0.3, 0)
    visualization[img_height:, :img_width] = overlay_img
    
    # Bottom-right: RGB image with polygon contours
    polygon_img = rgb_img.copy()
    for poly in polygons:
        # Convert normalized coordinates back to pixel coordinates
        points = []
        for i in range(0, len(poly), 2):
            x = int(poly[i] * img_width)
            y = int(poly[i + 1] * img_height)
            points.append([x, y])
        
        if len(points) >= 3:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(polygon_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Also fill the polygon with semi-transparent green
            polygon_overlay = np.zeros_like(polygon_img)
            cv2.fillPoly(polygon_overlay, [points], (0, 255, 0))
            polygon_img = cv2.addWeighted(polygon_img, 0.8, polygon_overlay, 0.2, 0)
    
    visualization[img_height:, img_width:] = polygon_img
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)
    
    # Add labels with black background for better visibility
    labels = ["Original Image", "Binary Mask", "Mask Overlay", "Polygon Contours"]
    positions = [(10, 30), (img_width + 10, 30), (10, img_height + 30), (img_width + 10, img_height + 30)]
    
    for label, pos in zip(labels, positions):
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        # Draw black background
        cv2.rectangle(visualization, (pos[0] - 5, pos[1] - text_height - 5), 
                     (pos[0] + text_width + 5, pos[1] + baseline + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(visualization, label, pos, font, font_scale, text_color, font_thickness)
    
    # Add object name at the top
    obj_label = f"Object: {obj_name}"
    (text_width, text_height), baseline = cv2.getTextSize(obj_label, font, font_scale + 0.2, font_thickness)
    cv2.rectangle(visualization, (fig_width // 2 - text_width // 2 - 10, 5), 
                 (fig_width // 2 + text_width // 2 + 10, text_height + baseline + 10), (0, 0, 0), -1)
    cv2.putText(visualization, obj_label, (fig_width // 2 - text_width // 2, text_height + 10), 
                font, font_scale + 0.2, (0, 255, 255), font_thickness)
    
    # Save visualization
    Image.fromarray(visualization).save(save_path)