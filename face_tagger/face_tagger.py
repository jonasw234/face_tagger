#!/usr/bin/env python3
"""Tries to recognize faces in input image list and update file’s metadata.

Usage:
gesichtserkennung.py [-v] [--tolerance=<float>] PATH ...

Options:
    -t <float>, --tolerance=<float> Tolerance for face detection (higher: more false
                                    positives, less false negatives) [default: 0.55].
    -v, --verbose                   Verbose output
"""
import glob
import logging
import os
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import face_recognition.api as face_recognition
import numpy as np
import PIL.Image
from docopt import docopt

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
TOP_KEYWORD = "Personen"


def exiftool_write(
    file: str, hierarchical_metadata: str = TOP_KEYWORD, metadata: str = TOP_KEYWORD
):
    """Use exiftool to write metadata.

    Parameters
    ----------
    file : str
        Path to the file
    hierarchical_metadata : str
        Hierarchical metadata to write
    metadata : str
        Metadata to write
    """
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-L",  # Don’t convert encodings
            "-charset",
            "filename=cp1252",  # For Windows file paths
            f"-Keywords+={metadata}",
            f"-HierarchicalSubject+={hierarchical_metadata}",
            f"-Subject+={metadata}",
            file,
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def scan_image_files(files: List[str]) -> List[str]:
    """Iterate through all files in a list of files.
    Recursively add image files to the list if the file is a directory.

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
    for file in files:
        if os.path.isdir(file):
            logging.debug(
                "Adding *.jpeg, *.jpg, and *.png files in %s directory "
                "(and its subdirectories) to input list.",
                file,
            )
            # Add images in directory
            image_extensions = ["jpeg", "jpg", "png"]
            for extension in image_extensions:
                files.extend(glob.glob(os.path.join(file, f"*.{extension}")))
                # Also from subdirectories
                files.extend(glob.glob(os.path.join(file, f"**/*.{extension}")))
            continue
        if not os.path.isfile(file):
            logging.warning("Cannot read %s, skipping ...", file)
            continue
        filenames.append(file)
    return filenames


def analyze_file(
    file: str, known_names: list, known_face_encodings: list, tolerance: float = 0.55
) -> List[str]:
    """Analyze a file and try to identify people in it.

    Parameters
    ----------
    file : str
        The file to analyze
    known_names : list
        List of known names
    known_face_encodings : list
        List of known face encodings
    tolerance : float
        Tolerance for determining whether two faces are the same (lower for
        less false positives and more false negatives)

    Returns
    -------
    List[str]
        The list of recognized people
    """
    try:
        unknown_image = face_recognition.load_image_file(file)
    except FileNotFoundError:
        logging.warning("%s was removed before processing.", file)
        return []
    # Scale down image if it’s giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.Resampling.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

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
        The name of the folder with known people

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

    known_people_files = scan_image_files([known_people_folder])
    for file in known_people_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        # Check cache first
        filesize_bytes = os.path.getsize(file)
        idx = known_names.index(basename) if basename in known_names else -1
        if idx != -1 and filesize_bytes == known_filesize_bytes[idx]:
            logging.debug("Using cached encodings for %s.", file)
            continue
        logging.debug("%s does not exist in the cache yet.", file)
        cache_dirty = True
        # Not found in cache
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            logging.debug(
                "More than one face found in %s. Only considering the first face.", file
            )
        elif not encodings:
            logging.debug("No faces found in %s. Ignoring file.", file)
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


def add_metadata(file: str, recognized_people: List[str]):
    """Add list of recognized people to a file.

    Parameters
    ----------
    file : str
        File to add the metadata to
    recognized_people : List[str]
        List of recognized people
    """
    # Found at least one recognized person
    current_metadata = (
        subprocess.run(["exiftool", "-Keywords", file], capture_output=True, check=True)
        .stdout.decode(encoding="ansi")
        .strip()
    )
    if not TOP_KEYWORD in current_metadata:
        # No person found before (assuming Bridge metadata format)
        exiftool_write(file)

    for person in recognized_people:
        if person not in current_metadata:
            logging.debug("Found %s in this image. Updating metadata!", person)
            # Newly found person
            exiftool_write(file, f"{TOP_KEYWORD}|{person}", person)
        else:
            logging.debug(
                "%s is already tagged in this image. Not adding them again.", person
            )


def main(files: list, tolerance: float = 0.55):
    """Read list of files, recognize faces and update metadata.

    Parameters
    ----------
    files : list
        List of input images
    tolerance : float
        Tolerance to use for face recognition
    """
    if not files:
        logging.critical("No files supplied to recognize!")
        return
    logging.info("Analyzing known faces for reference ...")
    known_names, known_face_encodings = scan_known_people(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "reference_images",
        ),
    )
    logging.debug("Using a tolerance of %f for face recognition.", tolerance)
    with ProcessPoolExecutor() as executor:
        futures = []
        for file in scan_image_files(files):
            logging.info("Starting analysis of %s ...", file)
            future = executor.submit(
                analyze_file, file, known_names, known_face_encodings, tolerance
            )
            futures.append(future)

        for future, file in zip(futures, scan_image_files(files)):
            recognized_people = future.result()
            logging.debug(
                "Analysis of %s finished. Proceeding adding recognized people to the metadata.",
                file,
            )
            if (
                recognized_people
                and "no_people_found" not in recognized_people
                and "warning:" not in recognized_people
            ):
                if recognized_people.count("unknown_person"):
                    logging.warning(
                        "%d unknown person(s) detected in %s. Please check manually!",
                        recognized_people.count("unknown_person"),
                        file,
                    )
                    while recognized_people.count(
                        "unknown_person"
                    ):  # Might be multiple
                        recognized_people.remove("unknown_person")
                if not recognized_people:
                    # Only unknown people were identified, so we can stop here
                    continue
                # Found at least one recognized person
                add_metadata(file, recognized_people)
            elif "warning:" in recognized_people:
                logging.warning("Warning during processing: %s", recognized_people)
            else:
                logging.debug(
                    "No person—not even someone unknown—identified in %s.", file
                )


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    main(args["PATH"], tolerance=float(args["--tolerance"]))
