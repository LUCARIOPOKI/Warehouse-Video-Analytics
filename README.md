# Project: Warehouse Video Analytics – Package Mishandling Detection

## Objectives / Goals
The goal of this project is to analyze warehouse CCTV footage and automatically detect instances of package mishandling. Examples of mishandling include:

- Throwing parcels
- Running or lying on the conveyor belt
- Kicking parcels

This will help improve operational efficiency, reduce damages, and enhance workplace safety.

---

## Work Completed So Far
- Implemented a **YOLOv9 model** to detect humans in the footage.
- Extracted video frames containing people and saved them separately.
- Split the extracted video into **10-second clips**.
- Processed these clips using Google’s **Gemini model** to identify potential mishandling activities.
- Added an exception to handle the model overloading and prevent execution from stopping midway.
- All the Mishandled video Urls will be stored in a text file.

---

## Challenges Faced & Solutions
- **Video Quality & Camera Coverage:**  
  Low resolution and limited coverage made detection harder.  
  *Solution:* Focused analysis on frames with people present to reduce noise.

- **System Limitations:**  
  Local system couldn’t efficiently process large video files.  
  *Solution:* Used Google Colab for higher compute capacity.

- **Model Overload Risk:**  
  Continuous processing caused the AI model to overload.  
  *Solution:* Adding a short waiting period before processing every 6th clip.

---

## Current Status
- The solution is **100% complete**.
- Detection of humans and mishandling activities is working effectively.
- Testing and fine-tuning are done.