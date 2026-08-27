from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from src.runtime import isolate_runtime
isolate_runtime()

import cv2
import numpy as np

from src.camera.camera import Camera
from src.runtime import load_yaml
from src.safety.path_guard import assert_safe_path,safe_mkdir,safe_open


def calibrate_from_observations(object_points,image_points,image_size):
    rms,matrix,distortion,rvecs,tvecs=cv2.calibrateCamera(
        object_points,image_points,image_size,None,None)
    errors=[]
    for objects,images,rvec,tvec in zip(object_points,image_points,rvecs,tvecs):
        projected,_=cv2.projectPoints(objects,rvec,tvec,matrix,distortion)
        errors.append(float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1,2)-images.reshape(-1,2))**2,axis=1)))))
    return {"rms":float(rms),"matrix":matrix,"distortion":distortion.reshape(-1),
            "per_view_rmse_px":errors}


def main():
    parser=argparse.ArgumentParser(description="Independent Phase 2A chessboard camera calibration")
    parser.add_argument("--columns",type=int,default=9,help="inner chessboard corners")
    parser.add_argument("--rows",type=int,default=6,help="inner chessboard corners")
    parser.add_argument("--square-size-mm",type=float,default=25.0)
    parser.add_argument("--minimum-samples",type=int,default=15)
    parser.add_argument("--output",default="configs/camera_intrinsics_phase2a.json")
    args=parser.parse_args()
    print("Required chessboard specification:")
    print(f"  inner corners: rows={args.rows}, columns={args.columns}")
    print(f"  measured square size: {args.square_size_mm:.3f} mm")
    print(f"  minimum valid images: {args.minimum_samples}")
    print("Capture front/left/right/up/down tilt plus near/far views; SPACE captures, C calibrates.")
    output=assert_safe_path(PROJECT_ROOT/args.output)
    original=(PROJECT_ROOT/"configs"/"camera_intrinsics.json").resolve()
    if output==original:
        raise PermissionError("Phase 2A refuses to overwrite the intrinsics file used by the original demo")
    config=load_yaml(PROJECT_ROOT/"configs/default.yaml"); camera_cfg=config["camera"]
    camera=Camera(camera_cfg["index"],camera_cfg["width"],camera_cfg["height"],camera_cfg["fps"])
    session=safe_mkdir(PROJECT_ROOT/"reports"/"camera_calibration"/
                       datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"))
    object_template=np.zeros((args.rows*args.columns,3),np.float32)
    object_template[:,:2]=np.mgrid[0:args.columns,0:args.rows].T.reshape(-1,2)
    object_template*=float(args.square_size_mm)/1000.0
    objects=[]; images=[]; last_corners=None; last_found=False; image_size=None
    try:
        camera.open(); ok,frame=camera.read()
        while ok:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); image_size=(gray.shape[1],gray.shape[0])
            found,corners=cv2.findChessboardCorners(gray,(args.columns,args.rows))
            if found:
                corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
                last_corners=corners; last_found=True
                cv2.drawChessboardCorners(frame,(args.columns,args.rows),corners,found)
            else: last_found=False
            cv2.putText(frame,f"Samples {len(images)}/{args.minimum_samples} | SPACE capture | C calibrate | Q quit",
                        (12,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2,cv2.LINE_AA)
            cv2.putText(frame,f"Board {args.columns}x{args.rows} inner corners | square {args.square_size_mm:.1f} mm | corners {'FOUND' if found else 'NOT FOUND'}",
                        (12,58),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0) if found else (0,120,255),2,cv2.LINE_AA)
            cv2.imshow("Phase 2A Camera Calibration",frame)
            key=cv2.waitKey(1)&0xff
            if key in (27,ord('q'),ord('Q')): break
            if key==32 and last_found:
                objects.append(object_template.copy()); images.append(last_corners.copy())
                success,encoded=cv2.imencode(".png",frame)
                if success: encoded.tofile(assert_safe_path(session/f"sample_{len(images):03d}.png"))
            if key in (ord('c'),ord('C')):
                if len(images)<args.minimum_samples: continue
                result=calibrate_from_observations(objects,images,image_size)
                matrix=result["matrix"]; distortion=result["distortion"].tolist()
                payload={"image_width":image_size[0],"image_height":image_size[1],
                    "fx":float(matrix[0,0]),"fy":float(matrix[1,1]),
                    "cx":float(matrix[0,2]),"cy":float(matrix[1,2]),
                    "dist_coeffs":distortion,"distortion":distortion,
                    "valid_calibration":True,"mode":"CALIBRATED_INTRINSICS",
                    "chessboard_inner_corners":[args.columns,args.rows],
                    "square_size_mm":args.square_size_mm,"sample_count":len(images),
                    "num_valid_images":len(images),
                    "timestamp":datetime.now(timezone.utc).isoformat(),
                    "calibration_rms":result["rms"],
                    "per_view_rmse_px":result["per_view_rmse_px"]}
                with safe_open(output,"w",encoding="utf-8") as handle:
                    json.dump(payload,handle,indent=2,ensure_ascii=False)
                with safe_open(session/"calibration_report.json","w",encoding="utf-8") as handle:
                    json.dump(payload,handle,indent=2,ensure_ascii=False)
                report=["# Phase 2A Camera Calibration","",
                    f"- Timestamp: {payload['timestamp']}",
                    f"- Chessboard inner corners: {args.columns} columns x {args.rows} rows",
                    f"- Square size: {args.square_size_mm} mm",
                    f"- Valid images: {len(images)}",
                    f"- Resolution: {image_size[0]} x {image_size[1]}",
                    f"- Calibration RMS: {result['rms']:.6f}",
                    f"- fx/fy: {payload['fx']:.6f} / {payload['fy']:.6f}",
                    f"- cx/cy: {payload['cx']:.6f} / {payload['cy']:.6f}",
                    f"- Distortion: {distortion}"]
                with safe_open(PROJECT_ROOT/"reports"/"camera_calibration_phase2a.md","w",encoding="utf-8") as handle:
                    handle.write("\n".join(report))
                print("Calibrated intrinsics written:",output); break
            ok,frame=camera.read()
    finally:
        camera.release(); cv2.destroyAllWindows()
    if not output.exists():
        print("PENDING_REAL_CAMERA_CALIBRATION; no intrinsics file was written")


if __name__=="__main__": main()
