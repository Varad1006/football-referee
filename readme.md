## 🗺️ Future Roadmap

*   **Multi-View Synchronization:** Implementing foul recognition across multiple camera angles simultaneously to handle occlusion and perspective distortion.
*   **FastAPI & Docker Deployment:** Containerizing the inference engine and exposing it via a FastAPI backend to separate the heavy compute from the Streamlit frontend.
*   **Offside Line Detection:** Expanding the ruleset engine to utilize pitch segmentation and vanishingHere is a comprehensive, professional `README.md` designed specifically to make this project stand out on your GitHub and resume. It highlights the complex engineering challenges you solved—like memory management and multi-model synchronization—which recruiters and senior engineers look for.

***
```markdown
# ⚽ DeepRef: AI Football Referee

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![YOLO](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)

DeepRef is an end-to-end, multi-modal computer vision pipeline that acts as an automated football referee. By synchronizing object detection (YOLO), player tracking (DeepSORT), and temporal deep learning models, the system evaluates player interactions, calculates physical collision severity, and determines if the ball was played cleanly to issue real-time disciplinary decisions (No Foul, Yellow Card, Red Card) based on FIFA Law 12.

## ✨ Key Features

*   **Multi-Model Architecture:** Combines spatial awareness (YOLOv8) with temporal severity classification (RGB & Grayscale Ensemble DL models) to accurately mimic human referee decision-making.
*   **True "Ball-First" Detection:** Replaces static heuristics with dynamic bounding box intersection math. The system calculates reach margins to accurately detect if an outstretched leg won the ball before peak collision.
*   **Memory-Optimized Pipeline:** Uses a custom ring-buffer for clip extraction, storing frames at a highly compressed 128x128 resolution. Frames are upscaled Just-In-Time (JIT) to 256x256 right before inference, preventing RAM bottlenecks during high-FPS video processing.
*   **Explainable AI UI:** A fully custom Streamlit frontend that breaks down the exact model outputs (Action Class, Collision Severity, Ball Touched First) and extracts contextual keyframes (2 seconds before, point of contact, 1.5 seconds after) to justify the final referee decision.

## 🧠 System Architecture

The pipeline processes video through four distinct stages:

1.  **Detection & Tracking (`detection.py` & `tracking.py`):** 
    *   A custom-trained YOLO model (`best.pt`) detects players and the ball. 
    *   DeepSORT assigns stable IDs to players across frames (with an IoU-based fallback tracker for environments without DeepSORT).
2.  **Interaction Detection (`interaction.py`):**
    *   Monitors Euclidean distances between tracked players.
    *   Determines if a collision event has occurred and calculates if either player's bounding box intersected with the ball prior to peak impact.
3.  **Clip Extraction & Upscaling (`clip_extraction.py` & `inference.py`):**
    *   Extracts an 8-frame temporal window around the peak collision.
    *   Applies JIT upscaling to feed precisely formatted data to the temporal classifiers.
4.  **Aggregation & Decision (`aggregation.py`):**
    *   Fuses predictions using a hybrid max/mean formula heavily weighted towards the peak collision frame.
    *   Applies the YOLO ball-touch override: if the ball was won cleanly, collision severity is suppressed, resulting in a "No Foul".

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/varadadhyapak/DeepRef.git
cd DeepRef
2. Create a virtual environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies

Bash
pip install -r requirements.txt
(Ensure you have ultralytics, tensorflow, opencv-python, streamlit, and deep-sort-realtime installed).

4. Add Model Weights
Place your custom trained weights in the root directory:

best.pt (YOLO custom weights)

model_1.h5 (RGB temporal classifier)

model_2.h5 (Grayscale temporal classifier)

5. Run the Application

Bash
streamlit run app.py
🗺️ Future Roadmap
Multi-View Synchronization: Implementing foul recognition across multiple camera angles simultaneously to handle occlusion and perspective distortion.

FastAPI & Docker Deployment: Containerizing the inference engine and exposing it via a FastAPI backend to separate the heavy compute from the Streamlit frontend.

Offside Line Detection: Expanding the ruleset engine to utilize pitch segmentation and vanishing point algorithms for automated offside calls.# football-referee
