from dotenv import load_dotenv, set_key
from google.genai import types
from google import genai
from pathlib import Path
import time
import json
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
folder_path = Path("onlyHumansFrame_10s_output_clips")  # folder containing the clips to be analyzed

def mishandling_detection(video_file_name):
    video_bytes = open(video_file_name, 'rb').read()

    print(f"processing video...{video_file_name}")
    response = client.models.generate_content(
        model='models/gemini-2.5-flash',
        config=types.GenerateContentConfig(
                system_instruction="""
                You are a video-analytics assistant for warehouse CCTV. Your task is to review the ENTIRE provided clip and determine whether any of the specific mishandling actions below occur. Return ONLY valid JSON (RFC 8259) with EXACT keys and boolean values; no extra text, no code fences, no trailing commas.

                ACTIONS TO DETECT (definitions & boundaries)
                1) parcel throwing → A person propels a parcel through the air with noticeable force so it leaves their hand(s) and travels ballistically. Distinguish from: gently placing/handing over, short drops under 0.5 m without force.
                2) parcel sliding → A person pushes a parcel across a surface so it maintains contact with that surface while moving. Distinguish from: rolling, carrying, lifting & placing.
                3) kicking/stepping on parcel → A foot makes deliberate contact with a parcel, either:
                   • kicking: striking the parcel with the foot to move or impact it, or
                   • stepping: placing body weight on the parcel (full or partial) while standing/walking.
                   Accidental light brushes or near-misses do NOT qualify.
                4) lying on the conveyor belts → A person’s torso and hips are in contact with the conveyor belts for ≥ 1 second. Kneeling or squatting is NOT lying.
                5) running on conveyor belts → A person runs (faster than a walking gait) with one or both feet on a MOVING conveyor belt. Running beside or across the belt (on the floor) does NOT qualify. Walking on a belt also qualifies.

                DECISION RULES
                - Mark an action true ONLY if the visual evidence is clear (≥ 0.8 confidence). If unclear, occluded, off-frame, or blurred: mark false.
                - Multiple actions can be true simultaneously if each meets the definition.
                - Consider intent and motion cues (arm swing, parcel trajectory, continuous surface contact, gait speed).
                - Ignore reflections, shadows, on-screen text overlays, timestamps, or UI elements.
                - Focus on human actions; actions by robots/machines/forklifts do NOT count as the listed actions.

                WHAT TO WRITE IN "Explanation"
                - Mention the cues you used (e.g., “ballistic arc,” “continuous surface contact,” “foot placed with weight,” “running gait on moving belt”).
                - If all are false, state the main reason (e.g., “only carrying and gentle placing observed”).

                OUTPUT FORMAT (STRICT JSON):
                Dont mention anything else other than the JSON object below.
                **STRICT** Dont mention "json" in your response.
                {
                  "Explanation": "brief explanation of the actions detected",
                  "has_parcel_throwing": true or false,
                  "has_parcel_sliding": true or false,
                  "has_kicking": true or false,
                  "has_lying": true or false,
                  "has_running": true or false
                }

                NOTES
                - Use lowercase true/false booleans.
                - Do not add keys or arrays. Do not include images or frame IDs.
                - If in doubt for any action, choose false and explain briefly why.

                """,
            temperature=0.1,
            # max_output_tokens=200
            ),
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text='Classify the actions in this video.'),
            ]
        )
    )

    print("\n--- RESPONSE ---\n")
    # print(type(response.text))
    # print(response.text)
    llm_response = response.text
    cleaned_response = llm_response.strip().removeprefix("json").strip()
    cleaned_response = cleaned_response.strip().removeprefix("json").strip()
    print("Cleaned response:", cleaned_response)
    dict_response = json.loads(cleaned_response) 
    if dict_response["has_parcel_throwing"] or dict_response["has_parcel_sliding"] or dict_response["has_kicking"] or dict_response["has_lying"] or dict_response["has_running"]:
        print("Mishandling Detected!\n")
        print(llm_response, "\n")
        with open("mishandling_detected.txt", "a") as f:
            f.write(f"{video_file_name}\n")
    else:
        print("No mishandling detected.\n")

# print(mishandling_detection("onlyHumansFrame_10s_output_clips/clip_1130_1140.mp4"))
    

if __name__ == "__main__":
    for file in folder_path.iterdir():
        if file.is_file():
            print("processing: ", file, "\n")
            mishandling_detection(str(file))
        print("--------------------------------------------------\n")
    print("\n --------------------Execution ended--------------------")


# print(response.usage_metadata)
# values = json.loads(response.text)
# print(values["has_parcel_throwing"])

# python gemini_object_detection.py
# ? 7 iterations before hitting The model is overloaded error

# Mishandling Detected clips
# throwing:
"10s_output_2ndclips/clip_40_50.mp4"
"10s_output_2ndclips/clip_50_60.mp4"
"10s_output_2ndclips/clip_90_100.mp4"
"10s_output_2ndclips/clip_140_150.mp4"
# Running:
"Conveyor_10s_output_clips/clip_1130_1140.mp4"

"""   You are a helpful assistant that analyzes CCTV videos to detect specific mishandling actions.  
Look at the video and identify if any of the following actions are being performed by individuals in the video.
Look very carefully and make sure you don't miss any action.
You have to classify the actions in the video to one of the following: 
    - parcel throwing
    - parcel sliding 
    - stepping 
    - lying 
    - running on conveyor belts. 

If an action is present in the video, mark it as true, otherwise false.
    - if the action is not clear, mark it as false.
    - If a person in the video is thwrowing a parcel, mark "has_parcel_throwing" as true.
    - If a person in the video is sliding a parcel, mark "has_parcel_sliding" as true.
    - If a person in the video is kicking or stepping on a parcel, mark "has_kicking" as true.
    - If a person in the video is lying on the ground, mark "has_lying" as true.
    - If a person in the video is running on conveyor belts, mark "has_running" as true.
    - If none of these actions are present, mark all as false.

Explain your reasoning briefly in the "Explanation" field.
Just Return the output as JSON 
in the following format:
`{
    "Explanation": "brief explanation of the actions detected",
    "has_parcel_throwing": true | false,
    "has_parcel_sliding": true | false", 
    "has_kicking": true | false, 
    "has_lying": true | false, 
    "has_running": true | false,
}`"""