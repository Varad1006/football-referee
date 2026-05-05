import cv2
import numpy as np
from pipeline import PipelineResult # Assuming this is where your result object lives

def export_annotated_highlight(video_path: str, result: PipelineResult, output_path: str = "output_highlight.mp4"):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: fps = 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use mp4v or avc1 codec (avc1 is usually better for web/Streamlit playback)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    peak_idx = result.aggregation.peak_frame_idx
    decision = result.aggregation.decision
    
    # How long the alert stays on screen (e.g., 2 seconds)
    alert_duration_frames = fps * 2 
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. DRAW ALERT OVERLAY
        # If we are at or just after the peak frame, trigger the foul alert
        if peak_idx <= frame_idx <= (peak_idx + alert_duration_frames):
            
            # Choose color based on the card given
            color = (0, 0, 255) if "Red" in decision.verdict else (0, 255, 255) if "Yellow" in decision.verdict else (0, 255, 0)
            
            # Draw a thick border around the whole video
            cv2.rectangle(frame, (0, 0), (width, height), color, thickness=15)
            
            # Add the Alert Text
            text = f"FOUL DETECTED: {decision.verdict.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (50, 100), font, 2, (0, 0, 0), 8) # Black outline
            cv2.putText(frame, text, (50, 100), font, 2, color, 4)     # Colored text inside
            
        # 2. DRAW BOUNDING BOXES 
        # Note: You will need to pass your tracking history into this function, 
        # or re-run the tracker here if it isn't saved in memory.
        # Example if you had a dictionary of tracks per frame: tracks_per_frame[frame_idx]
        """
        if frame_idx in tracks_per_frame:
            for player in tracks_per_frame[frame_idx]:
                x1, y1, x2, y2 = map(int, player.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f"ID: {player.track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        """

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    return output_path