"""Perspective extraction from equirectangular panos (py360convert). Used by reproject.py."""
import cv2
import numpy as np
import py360convert

def extract_left_right_views(
    pano_path: str,
    heading_deg: float,
    fov_h_deg: float = 120.0,
    fov_v_deg: float = 90.0,
    pitch_deg: float = 10.0,
    out_width: int = 1024,
    out_height: int = 768,
):
    """
    pano_path   : chemin vers l'image 360° équirectangulaire (2:1)
    heading_deg : orientation du véhicule en degrés (0 = nord, 90 = est, etc.)
                  -> typiquement computed_compass_angle de Mapillary
    fov_h_deg   : FOV horizontal de la vue perspective (°)
    fov_v_deg   : FOV vertical (°)
    pitch_deg   : pitch de la caméra virtuelle, positif = on regarde un peu vers le haut
    out_width   : largeur de l'image de sortie
    out_height  : hauteur de l'image de sortie
    """

    # 1) Charger l'image pano (BGR -> RGB)
    pano_bgr = cv2.imread(pano_path, cv2.IMREAD_COLOR)
    if pano_bgr is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {pano_path}")
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

    # 2) FOV (horizontal, vertical)
    fov = (fov_h_deg, fov_v_deg)

    def view_at_offset(offset_deg: float):
        """
        offset_deg : +90 = côté droit du véhicule, -90 = côté gauche
        """
        # yaw en degrés, -Left / +Right (convention py360convert)
        yaw = (heading_deg + offset_deg) % 360

        # py360convert.e2p :
        # e_img, fov_deg, u_deg (yaw), v_deg (pitch), out_hw, in_rot_deg, mode
        persp = py360convert.e2p(
            pano_rgb,
            fov_deg=fov,
            u_deg=yaw,
            v_deg=pitch_deg,          # pitch > 0 = on relève un peu la vue
            out_hw=(out_height, out_width),
            in_rot_deg=0,
            mode="bilinear",
        )
        # repasser en BGR pour cv2.imwrite
        return cv2.cvtColor(persp, cv2.COLOR_RGB2BGR)

    # 3) Côté droit et côté gauche par rapport au sens de marche
    right_bgr = view_at_offset(+90.0)
    left_bgr  = view_at_offset(-90.0)

    return left_bgr, right_bgr
