from dotenv import load_dotenv
from google.genai import types
from google import genai
from pathlib import Path
import time
import json
import os
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
folder_path = Path("onlyHumansFrame_throwing_positive_10s_output_clips")  # folder containing the 10 sec clips to be analyzed

def safe_parse_json(llm_response: str):
    if not llm_response or not llm_response.strip():
        print("⚠️ Empty response from LLM!")
        return {}

    match = re.search(r"\{.*\}", llm_response, re.DOTALL)
    if not match:
        print("⚠️ No JSON found in response!")
        print("Raw response:", llm_response)
        return {}

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("⚠️ JSON decode failed:", e)
        print("Raw JSON candidate:", json_str)
        return {}


def mishandling_detection(video_file_name):
    video_bytes = open(video_file_name, 'rb').read()

    # print(f"processing video...{video_file_name}")
    response = client.models.generate_content(
        model='models/gemini-2.5-flash',
        config=types.GenerateContentConfig(
                system_instruction="""
                You are a video-analytics assistant for warehouse CCTV. Your task is to review the ENTIRE provided clip and determine whether any of the specific mishandling actions below occur. Return ONLY valid JSON (RFC 8259) with EXACT keys and boolean values; no extra text, no code fences, no trailing commas.

                ACTIONS TO DETECT (definitions & boundaries)
                1) parcel throwing → A person propels a parcel through the air with noticeable force so it leaves their hand(s) and travels ballistically. Distinguish from: gently placing/handing over, short drops under 0.5 m without force.
                2) kicking/stepping on parcel → A foot makes deliberate contact with a parcel, either:
                   • kicking: striking the parcel with the foot to move or impact it, or
                   • stepping: placing body weight on the parcel (full or partial) while standing/walking.
                   Accidental light brushes or near-misses do NOT qualify.
                3) lying on the conveyor belts → A person’s torso and hips are in contact with the conveyor belts for ≥ 1 second. Kneeling or squatting is NOT lying.
                4) running on conveyor belts → A person runs (faster than a walking gait) with one or both feet on a MOVING conveyor belt. Running beside or across the belt (on the floor) does NOT qualify. Walking on a belt also qualifies.

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
                **STRICT**: Dont mention "json" name in your response.
                
                {
                  "Explanation": "brief explanation of the actions detected",
                  "has_parcel_throwing": true or false,
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
    llm_response = response.text
    dict_response = safe_parse_json(llm_response)
    if dict_response["has_parcel_throwing"]  or dict_response["has_kicking"] or dict_response["has_lying"] or dict_response["has_running"]:
        print("Mishandling Detected!\n")
        print(llm_response, "\n")
        with open("mishandling_detected.txt", "a") as f:
            f.write(f"{video_file_name}\n{llm_response}\n\n")
    else:
        print("No mishandling detected.\n")

if __name__ == "__main__":
    start_time = time.time()
    iterations = 0
    
    with open("mishandling_detected.txt", "a") as f:
            f.write(f"------------------------files from {folder_path} folder--------------------------\n")
    
    for file in folder_path.iterdir():
        iterations += 1
        
        if file.is_file():
            print(f"File No: {iterations} ")
            print("processing: ", file)
            mishandling_detection(str(file))

        if iterations % 3 == 0:
            print("Taking a 10 seconds break to avoid The model is overloaded error...\n")
            time.sleep(10)

        print("--------------------------------------------------\n")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
    print("\n --------------------Execution ended--------------------")

# python gemini_object_detection.py

# Throwing Negative:
# Time to extract humans: 20 mins 
# Time to split the clip for 10s each: 3 mins
# Time to analyze the clips: 22 mins # Total: 45 mins 

# conveyor negative 
# Time to extract humans: 17 mins 
# Time to split the clip for 10s each: 3 mins
# Time to analyze the clips: 20 mins # Total: 40 mins 

# Throwing Positive:
# Time to extract humans: 10 mins
# Time to split the clip for 10s each: 2 mins
# Time to analyze the clips: 15 mins # Total: 27 mins