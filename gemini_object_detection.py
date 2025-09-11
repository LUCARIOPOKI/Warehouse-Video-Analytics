"""
Production-grade video mishandling detection system for warehouse CCTV analysis.

This module provides robust video analysis capabilities with comprehensive error handling,
logging, monitoring, and retry mechanisms for detecting mishandling actions in warehouse
surveillance footage.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from dotenv import load_dotenv
from google.genai import types
from google import genai


@dataclass
class ProcessingStats:
    """Statistics for video processing session."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    mishandling_detected: int = 0
    start_time: float = 0
    end_time: float = 0
    
    @property
    def duration(self) -> float:
        """Total processing duration in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0


@dataclass
class MishandlingResult:
    """Result of mishandling detection for a single video."""
    file_path: str
    has_parcel_throwing: bool = False
    has_kicking: bool = False
    has_lying: bool = False
    has_running: bool = False
    explanation: str = ""
    processing_time: float = 0
    error: Optional[str] = None
    
    @property
    def has_any_mishandling(self) -> bool:
        """Check if any mishandling action was detected."""
        return any([self.has_parcel_throwing, self.has_kicking, self.has_lying, self.has_running])


class VideoMishandlingDetector:
    """
    Production-grade video mishandling detection system.
    
    Features:
    - Comprehensive error handling and retry logic
    - Structured logging with multiple levels
    - Progress tracking and statistics
    - Configurable processing parameters
    - Graceful degradation on failures
    """
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    MAX_RETRIES = 3
    RETRY_DELAY = 3.0
    BATCH_DELAY = 10.0
    
    def __init__(
        self,
        folder_path: str,
        output_file: str = "mishandling_detected.txt",
        log_level: str = "INFO",
        batch_size: int = 3
    ):
        """
        Initialize the video mishandling detector.
        
        Args:
            folder_path: Path to folder containing video files
            output_file: Path to output file for detected mishandling
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            batch_size: Number of files to process before taking a break
        """
        self.folder_path = Path(folder_path)
        self.output_file = Path(output_file)
        self.batch_size = batch_size
        self.stats = ProcessingStats()
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Initialize API client
        self.client = self._initialize_client()
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized VideoMishandlingDetector for folder: {self.folder_path}")
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup structured logging with file and console handlers."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler for detailed logs
        log_file = Path(f"video_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for user-friendly output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def _initialize_client(self) -> genai.Client:
        """Initialize and validate the Gemini API client."""
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            client = genai.Client(api_key=api_key)
            self.logger.info("Gemini API client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API client: {e}")
            raise
    
    def _safe_parse_json(self, llm_response: str) -> Dict:
        """
        Safely parse JSON from LLM response with comprehensive error handling.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Parsed JSON dictionary or empty dict on failure
        """
        if not llm_response or not llm_response.strip():
            self.logger.warning("Empty response from LLM")
            return {}
        
        # Try to extract JSON using regex
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Nested braces
            r"\{.*\}",  # Simple extraction
        ]
        
        json_str = None
        for pattern in json_patterns:
            match = re.search(pattern, llm_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                break
        
        if not json_str:
            self.logger.warning(f"No JSON found in response: {llm_response[:200]}...")
            return {}
        
        # Try to parse JSON with multiple attempts
        for attempt in range(2):
            try:
                parsed = json.loads(json_str)
                self.logger.debug("Successfully parsed JSON response")
                return parsed
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON decode attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    # Try cleaning the JSON string
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        self.logger.error(f"Failed to parse JSON after all attempts: {json_str[:200]}...")
        return {}
    
    @contextmanager
    def _error_handling(self, video_path: str, operation: str):
        """Context manager for consistent error handling."""
        try:
            self.logger.debug(f"Starting {operation} for {video_path}")
            yield
            
        except Exception as e:
            error_msg = f"Error during {operation} for {video_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise
    
    def _analyze_video_with_retry(self, video_path: Path) -> MishandlingResult:
        """
        Analyze video with retry logic and comprehensive error handling.
        
        Args:
            video_path: Path to video file
            
        Returns:
            MishandlingResult with detection results
        """
        result = MishandlingResult(file_path=str(video_path))
        start_time = time.time()
        
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._error_handling(str(video_path), f"analysis attempt {attempt + 1}"):
                    # Read video file
                    if not video_path.exists():
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                    
                    if video_path.stat().st_size == 0:
                        raise ValueError(f"Video file is empty: {video_path}")
                    
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    self.logger.debug(f"Read {len(video_bytes)} bytes from {video_path}")
                    
                    # Make API call
                    response = self.client.models.generate_content(
                        model='models/gemini-2.5-flash',
                        config=types.GenerateContentConfig(
                            system_instruction=self._get_system_instruction(),
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
                    
                    # Parse response
                    llm_response = response.text
                    self.logger.debug(f"Received response: {llm_response[:100]}...")
                    
                    parsed_response = self._safe_parse_json(llm_response)
                    
                    if not parsed_response:
                        raise ValueError("Failed to parse valid JSON from response")
                    
                    # Update result with parsed data
                    result.has_parcel_throwing = parsed_response.get("has_parcel_throwing", False)
                    result.has_kicking = parsed_response.get("has_kicking", False)
                    result.has_lying = parsed_response.get("has_lying", False)
                    result.has_running = parsed_response.get("has_running", False)
                    result.explanation = parsed_response.get("Explanation", "")
                    result.processing_time = time.time() - start_time
                    
                    self.logger.info(
                        f"Successfully analyzed {video_path.name} "
                        f"(attempt {attempt + 1}, {result.processing_time:.2f}s)"
                    )
                    
                    return result
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed for {video_path}: {str(e)}"
                self.logger.warning(error_msg)
                
                # Check if it's a model overload error
                if "overloaded" in str(e).lower() or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    self.logger.info(f"Model overload detected, waiting {self.RETRY_DELAY} seconds...")
                    time.sleep(self.RETRY_DELAY)
                elif attempt < self.MAX_RETRIES - 1:
                    time.sleep(1)  # Short delay for other errors
        
        # All attempts failed
        result.error = f"Failed after {self.MAX_RETRIES} attempts"
        result.processing_time = time.time() - start_time
        self.logger.error(f"All attempts failed for {video_path}")
        return result
    
    def _get_system_instruction(self) -> str:
        """Get the system instruction for the LLM."""
        return """
        You are a video-analytics assistant for warehouse CCTV. Your task is to review the ENTIRE provided clip and determine whether any of the specific mishandling actions below occur. Return ONLY valid JSON (RFC 8259) with EXACT keys and boolean values; no extra text, no code fences, no trailing commas.

        ACTIONS TO DETECT (definitions & boundaries)
        1) parcel throwing → A person propels a parcel through the air with noticeable force so it leaves their hand(s) and travels ballistically. Distinguish from: gently placing/handing over, short drops under 0.5 m without force.
        2) kicking/stepping on parcel → A foot makes deliberate contact with a parcel, either:
           • kicking: striking the parcel with the foot to move or impact it, or
           • stepping: placing body weight on the parcel (full or partial) while standing/walking.
           Accidental light brushes or near-misses do NOT qualify.
        3) lying on the conveyor belts → A person's torso and hips are in contact with the conveyor belts for ≥ 1 second. Kneeling or squatting is NOT lying.
        4) running on conveyor belts → A person runs (faster than a walking gait) with one or both feet on a MOVING conveyor belt. Running beside or across the belt (on the floor) does NOT qualify. Walking on a belt also qualifies.

        DECISION RULES
        - Mark an action true ONLY if the visual evidence is clear (≥ 0.8 confidence). If unclear, occluded, off-frame, or blurred: mark false.
        - Multiple actions can be true simultaneously if each meets the definition.
        - Consider intent and motion cues (arm swing, parcel trajectory, continuous surface contact, gait speed).
        - Ignore reflections, shadows, on-screen text overlays, timestamps, or UI elements.
        - Focus on human actions; actions by robots/machines/forklifts do NOT count as the listed actions.

        WHAT TO WRITE IN "Explanation"
        - Mention the cues you used (e.g., "ballistic arc," "continuous surface contact," "foot placed with weight," "running gait on moving belt").
        - If all are false, state the main reason (e.g., "only carrying and gentle placing observed").

        OUTPUT FORMAT (STRICT JSON):
        Do not mention anything else other than the JSON object below.
        **STRICT**: Do not mention "json" name in your response.
        
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
        """
    
    def _get_video_files(self) -> List[Path]:
        """Get list of supported video files from the folder."""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        
        video_files = []
        for file_path in self.folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS:
                video_files.append(file_path)
        
        video_files.sort()  # Process in consistent order
        self.logger.info(f"Found {len(video_files)} video files to process")
        return video_files
    
    def _write_result_to_file(self, result: MishandlingResult) -> None:
        """Write detection result to output file."""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                if result.has_any_mishandling:
                    f.write(f"{result.file_path}\n")
                    f.write(f"Mishandling detected: Throwing={result.has_parcel_throwing}, "
                           f"Kicking={result.has_kicking}, Lying={result.has_lying}, "
                           f"Running={result.has_running}\n")
                    f.write(f"Explanation: {result.explanation}\n")
                    f.write(f"Processing time: {result.processing_time:.2f}s\n\n")
                    
        except Exception as e:
            self.logger.error(f"Failed to write result to file: {e}")
    
    def _log_progress(self, current: int, total: int, result: MishandlingResult) -> None:
        """Log processing progress."""
        progress_pct = (current / total * 100) if total > 0 else 0
        status = "MISHANDLING" if result.has_any_mishandling else "CLEAN"
        
        if result.error:
            status = "ERROR"
        
        self.logger.info(
            f"Progress: {current}/{total} ({progress_pct:.1f}%) - "
            f"File: {Path(result.file_path).name} - Status: {status} - "
            f"Time: {result.processing_time:.2f}s"
        )
    
    def process_videos(self) -> ProcessingStats:
        """
        Process all videos in the folder and detect mishandling actions.
        
        Returns:
            ProcessingStats with processing statistics
        """
        self.stats.start_time = time.time()
        
        try:
            # Get video files
            video_files = self._get_video_files()
            self.stats.total_files = len(video_files)
            
            if self.stats.total_files == 0:
                self.logger.warning("No video files found to process")
                return self.stats
            
            # Write header to output file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Processing session started: {datetime.now().isoformat()}\n")
                f.write(f"Folder: {self.folder_path}\n")
                f.write(f"Files to process: {self.stats.total_files}\n")
                f.write(f"{'='*80}\n\n")
            
            # Process each video file
            for i, video_file in enumerate(video_files, 1):
                self.logger.info(f"Processing file {i}/{self.stats.total_files}: {video_file.name}")
                
                # Analyze video
                result = self._analyze_video_with_retry(video_file)
                
                # Update statistics
                if result.error:
                    self.stats.failed_files += 1
                else:
                    self.stats.processed_files += 1
                    if result.has_any_mishandling:
                        self.stats.mishandling_detected += 1
                
                # Write result and log progress
                self._write_result_to_file(result)
                self._log_progress(i, self.stats.total_files, result)
                
                # Take break after processing batch
                if i % self.batch_size == 0 and i < self.stats.total_files:
                    self.logger.info(f"Taking {self.BATCH_DELAY} second break after processing {i} files...")
                    time.sleep(self.BATCH_DELAY)
            
            self.stats.end_time = time.time()
            self._log_final_statistics()
            
        except Exception as e:
            self.logger.error(f"Fatal error during processing: {e}", exc_info=True)
            self.stats.end_time = time.time()
            raise
        
        return self.stats
    
    def _log_final_statistics(self) -> None:
        """Log final processing statistics."""
        self.logger.info("="*80)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Total files: {self.stats.total_files}")
        self.logger.info(f"Successfully processed: {self.stats.processed_files}")
        self.logger.info(f"Failed: {self.stats.failed_files}")
        self.logger.info(f"Mishandling detected: {self.stats.mishandling_detected}")
        self.logger.info(f"Success rate: {self.stats.success_rate:.1f}%")
        self.logger.info(f"Total duration: {self.stats.duration:.2f} seconds")
        self.logger.info(f"Average per file: {self.stats.duration/self.stats.total_files:.2f}s")
        
        # Write final statistics to output file
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Processing session completed: {datetime.now().isoformat()}\n")
                f.write(f"Total files: {self.stats.total_files}\n")
                f.write(f"Successfully processed: {self.stats.processed_files}\n")
                f.write(f"Failed: {self.stats.failed_files}\n")
                f.write(f"Mishandling detected: {self.stats.mishandling_detected}\n")
                f.write(f"Success rate: {self.stats.success_rate:.1f}%\n")
                f.write(f"Total duration: {self.stats.duration:.2f} seconds\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to write final statistics: {e}")


def main():
    """Main entry point for the video mishandling detection system."""
    try:
        # Configuration
        config = {
            "folder_path": "onlyHumansFrame_throwing_positive_10s_output_clips",
            "output_file": "mishandling_detected.txt",
            "log_level": "INFO",
            "batch_size": 3
        }
        
        # Initialize and run detector
        detector = VideoMishandlingDetector(**config)
        stats = detector.process_videos()
        
        # Return exit code based on results
        if stats.failed_files == 0:
            print(f"\n✅ All {stats.total_files} files processed successfully!")
            return 0
        elif stats.processed_files > 0:
            print(f"\n⚠️  Partial success: {stats.processed_files}/{stats.total_files} files processed")
            return 1
        else:
            print(f"\n❌ Processing failed for all files")
            return 2
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error("Fatal error in main", exc_info=True)
        return 3


if __name__ == "__main__":
    exit(main())

# python gemini_object_detection.py