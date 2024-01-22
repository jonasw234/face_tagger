#!/usr/bin/env python3
"""Tries to recognize faces in input image list and update file’s metadata.

Usage:
gesichtserkennung.py [-v] --references=<PATH> [--tolerance=<VALUE>] <PATH>...

-r PATH, --references <PATH>   Path to the folder with reference images
-t VALUE, --tolerance <VALUE>  Tolerance for face detection (higher: more false
                               positives, less false negatives) [default: 0.55].
-v, --verbose                  Verbose output
"""
import glob
import logging
import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import cv2
import face_recognition.api as face_recognition
import numpy as np
import PIL.Image
from docopt import docopt


IMAGE_EXTENSIONS = ["jpeg", "jpg", "png"]
VIDEO_EXTENSIONS = ["mp4", "avi", "mkv", "mov"]
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
TOP_KEYWORD = "Personen"


def exiftool_write(
    file_path: str,
    hierarchical_metadata: List[str] = [TOP_KEYWORD],
    metadata: List[str] = [TOP_KEYWORD],
):
    """Use exiftool to write metadata.

    Parameters
    ----------
    file_path : str
        Path to the file
    hierarchical_metadata : List[str]
        Hierarchical metadata to write
    metadata : List[str]
        Metadata to write
    """
    parameters = [
        "exiftool",
        "-overwrite_original",
        "-L",  # Don’t convert encodings
        "-charset",
        "filename=cp1252",  # For Windows file paths
    ]
    parameters.extend([f"-Keywords+={person}" for person in metadata])
    parameters.extend(
        [f"-HierarchicalSubject+={person_sub}" for person_sub in hierarchical_metadata]
    )
    parameters.extend([f"-Subject+={person}" for person in metadata])
    parameters.append(file_path)
    subprocess.run(
        parameters,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def scan_files(files: List[str]) -> List[str]:
    """Iterate through all files in a list of files.
    Recursively add image and movie files to the list if the file is a directory.

    Parameters
    ----------
    files : List[str]
        List of files

    Returns
    -------
    List[str]
        List of filenames
    """
    filenames = []
    for file_path in files:
        if os.path.isdir(file_path):
            # Add images and videos in directory
            logger.debug(
                "Adding %s files in %s directory "
                "(and its subdirectories) to input list.",
                f"{', '.join(ALLOWED_EXTENSIONS[:-1])}, and {ALLOWED_EXTENSIONS[-1]}",
                file_path,
            )
            for extension in ALLOWED_EXTENSIONS:
                files.extend(glob.glob(os.path.join(file_path, f"*.{extension}")))
                # Also from subdirectories
                files.extend(glob.glob(os.path.join(file_path, f"**/*.{extension}")))
            continue
        if not os.path.isfile(file_path):
            logger.warning("Cannot read %s, skipping ...", file_path)
            continue
        filenames.append(file_path)
    return filenames


def analyze_file(
    file_path: str,
    known_names: List[str],
    known_face_encodings: List[np.ndarray],
    tolerance: float = 0.55,
) -> List[str]:
    """Analyze a file and try to identify people in it.

    Parameters
    ----------
    file_path : str
        The file to analyze
    known_names : List[str]
        List of known names
    known_face_encodings : List[np.ndarray]
        List of known face encodings
    tolerance : float
        Tolerance for determining whether two faces are the same (lower for
        less false positives and more false negatives)

    Returns
    -------
    List[str]
        The list of recognized people
    """
    if file_path.lower().endswith(tuple(VIDEO_EXTENSIONS)):
        # Video file detected, analyze the first frame
        try:
            video_capture = cv2.VideoCapture(file_path)
            ret, frame = video_capture.read()
            video_capture.release()
            if not ret:
                logger.warning("Failed to read the first frame from %s", file_path)
                return []
        except Exception as e:
            logger.warning("Error processing video file %s: %s", file_path, str(e))
            return []
    else:
        # Regular image file
        try:
            frame = face_recognition.load_image_file(file_path)
        except FileNotFoundError:
            logger.warning("%s was removed before processing.", file_path)
            return []

    # Scale down image if it’s giant so things run a little faster
    if max(frame.shape) > 1600:
        pil_img = PIL.Image.fromarray(frame)
        pil_img.thumbnail((1600, 1600), PIL.Image.Resampling.LANCZOS)
        frame = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(frame)

    recognized_people = []

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(
            known_face_encodings, unknown_encoding
        )
        result = list(distances <= tolerance)

        if True in result:
            recognized_people.extend(
                [name for is_match, name in zip(result, known_names) if is_match]
            )
        else:
            recognized_people.append("unknown_person")

    if not unknown_encodings:
        # print out fact that no faces were found in image
        recognized_people.append("no_people_found")

    return recognized_people


def scan_known_people(known_people_folder: str) -> Tuple[List[str], List[np.ndarray]]:
    """Scan a folder with known people and return their name and encoding.

    Parameters
    ----------
    known_people_folder : str
        The path of the folder with known people

    Returns
    -------
    Tuple[List[str], List[np.ndarray]]
        Name of the person and encoding
    """
    known_names = []
    known_face_encodings = []

    # WARNING: Cache file needs to be trusted
    known_filesize_bytes = []
    cachefile_path = os.path.join(known_people_folder, "cache.pkl")
    cache_dirty = False

    try:
        with open(cachefile_path, "rb") as cachefile:
            known_names, known_face_encodings, known_filesize_bytes = pickle.load(
                cachefile
            )
    except FileNotFoundError:
        # No cache exists yet
        pass

    known_people_files = scan_files([known_people_folder])
    for file_path in known_people_files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        # Check cache first
        filesize_bytes = os.path.getsize(file_path)
        idx = known_names.index(basename) if basename in known_names else -1
        if idx != -1 and filesize_bytes == known_filesize_bytes[idx]:
            logger.debug("Using cached encodings for %s.", file_path)
            continue
        logger.debug("%s does not exist in the cache yet.", file_path)
        cache_dirty = True
        # Not found in cache
        img = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            logger.debug(
                "More than one face found in %s. Only considering the first face.",
                file_path,
            )
        elif not encodings:
            logger.debug("No faces found in %s. Ignoring file.", file_path)
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])
            known_filesize_bytes.append(filesize_bytes)
    if cache_dirty or len(known_people_files) != len(
        known_names
    ):  # Clear cache if changed
        # Update cache with new information
        with open(cachefile_path, "wb") as cachefile:
            pickle.dump(
                [known_names, known_face_encodings, known_filesize_bytes], cachefile
            )

    return known_names, known_face_encodings


def add_metadata(file_path: str, recognized_people: List[str]):
    """Add list of recognized people to a file.

    Parameters
    ----------
    file_path : str
        File to add the metadata to
    recognized_people : List[str]
        List of recognized people
    """
    recognized_people_sub = [f"{TOP_KEYWORD}|{person}" for person in recognized_people]
    recognized_people.append(TOP_KEYWORD)
    recognized_people_sub.append(TOP_KEYWORD)
    logger.debug("Adding the following metadata: %s", recognized_people)
    exiftool_write(file_path, recognized_people_sub, recognized_people)


def main(files: List[str], tolerance: float = 0.55, references: str = ""):
    """Read list of files, recognize faces and update metadata.

    Parameters
    ----------
    files : List[str]
        List of input images
    tolerance : float
        Tolerance to use for face recognition
    references : str
        Path to a folder with known people
    """
    if not files:
        logger.critical("No files supplied to recognize!")
        return
    logger.info("Analyzing known faces for reference ...")
    known_names, known_face_encodings = scan_known_people(references)
    logger.debug("Using a tolerance of %f for face recognition.", tolerance)
    with ProcessPoolExecutor() as executor:
        futures = []
        for file_path in scan_files(files):
            logger.info("Starting analysis of %s ...", file_path)
            future = executor.submit(
                analyze_file, file_path, known_names, known_face_encodings, tolerance
            )
            futures.append(future)

        for future, file_path in zip(futures, scan_files(files)):
            recognized_people = future.result()
            logger.debug(
                "Analysis of %s finished. Proceeding adding recognized people to the "
                "metadata.",
                file_path,
            )
            if (
                recognized_people
                and "no_people_found" not in recognized_people
                and "warning:" not in recognized_people
            ):
                if recognized_people.count("unknown_person"):
                    logger.warning(
                        "%d unknown person(s) detected in %s. Please check manually!",
                        recognized_people.count("unknown_person"),
                        file_path,
                    )
                    while recognized_people.count(
                        "unknown_person"
                    ):  # Might be multiple
                        recognized_people.remove("unknown_person")
                if not recognized_people:
                    # Only unknown people were identified, so we can stop here
                    continue
                # Found at least one recognized person
                add_metadata(file_path, recognized_people)
            elif "warning:" in recognized_people:
                logger.warning("Warning during processing: %s", recognized_people)
            else:
                logger.debug(
                    "No person—not even someone unknown—identified in %s.", file_path
                )


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--verbose"]:
        logger.setLevel(logging.DEBUG)
    main(
        args["<PATH>"],
        tolerance=float(args["--tolerance"]),
        references=args["--references"],
    )
