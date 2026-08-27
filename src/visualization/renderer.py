from __future__ import annotations

import cv2
import numpy as np

from src.geometry.backprojection import point_metrics

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def _text(frame, value, pos, color=(255,255,255), scale=0.52, thickness=1):
    cv2.putText(frame, str(value), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(frame, str(value), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(frame, result, fps, ray_length_px=900):
    hand = result.hand
    if hand.detected:
        pts = hand.landmarks_2d[:, :2].astype(int)
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (100,220,100), 2)
        for i, p in enumerate(pts):
            cv2.circle(frame, tuple(p), 5 if i in (5,6,7,8) else 3, (0,220,255) if i in (5,6,7,8) else (80,255,80), -1)
        if np.linalg.norm(hand.direction_2d) > 0:
            end = hand.index_tip + hand.direction_2d * ray_length_px
            cv2.arrowedLine(frame, tuple(hand.index_tip.astype(int)), tuple(end.astype(int)), (30,30,255), 3, tipLength=0.04)
        _text(frame, f"Gesture: {'POINTING' if hand.is_pointing else 'Hand'} ({hand.confidence:.2f})", (12, 88), (0,255,255))
        if result.tip_3d is not None:
            x,y,z = result.tip_3d
            metrics = point_metrics(result.tip_3d)
            _text(frame, f"TIP XYZ=({x:+.2f},{y:+.2f},{z:.2f})m D={metrics['distance']:.2f} H={metrics['height']:+.2f}", (12,112))
    else:
        _text(frame, "No Hand", (12, 88), (0,180,255))
    for cup in result.cups:
        x1,y1,x2,y2 = cup.bbox
        selected = cup.id == result.selected_cup
        color = (0,255,255) if selected else (255,150,30)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 4 if selected else 2)
        depth = f"Z={cup.depth:.2f}m" if cup.depth is not None else "Z=N/A"
        _text(frame, f"Cup {cup.id} {cup.confidence:.2f} {depth}", (x1, max(20,y1-8)), color)
        if cup.center_3d is not None:
            x,y,z = cup.center_3d
            _text(frame, f"XYZ {x:+.2f},{y:+.2f},{z:.2f}", (x1, min(frame.shape[0]-8,y2+20)), color, .43)
    selected = f"Cup {result.selected_cup}" if result.selected_cup is not None else "None"
    cv2.rectangle(frame, (0,0), (frame.shape[1], 62), (20,20,20), -1)
    _text(frame, f"POINTING: {selected}", (12,35), (0,255,255), .9, 2)
    calibration = result.diagnostics.get("calibration", "UNKNOWN")
    _text(frame, f"FPS {fps:.1f} | Depth: {result.depth_mode} | Camera: {calibration}", (12,58), (220,220,220), .45)
    if result.diagnostics.get("init_errors"):
        _text(frame, "Some models are in fallback/unavailable mode - see console", (12, frame.shape[0]-12), (0,160,255), .5)
    return frame

