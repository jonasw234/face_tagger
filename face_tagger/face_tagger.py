#!/usr/bin/env python3
"""Tries to recognize faces in input image list and update file’s metadata.

Usage:
gesichtserkennung.py [-v] [--tolerance=<float>] PATH ...

Options:
    -t <float>, --tolerance=<float> Tolerance for face detection (higher: more false
                                    positives, less false negatives) [default: 0.55].
    -v, --verbose                   Verbose output
"""
# TODO Make use of multithreading (prework already done below)
import glob
import itertools
import logging
import multiprocessing
import os
import pickle
import subprocess
import sys
from typing import Tuple

from docopt import docopt
import face_recognition.api as face_recognition
import numpy as np
import PIL.Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def exiftool_write(file: str, field: str, metadata: str):
    """Use exiftool to write metadata.

    Parameters
    ----------
    file : str
        Path to the file
    field : str
        Metadata field to append to
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
            f"-{field}+={metadata}",
            file,
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )


def scan_image_files(files: list) -> list:
    """Iterate through all files in a list of files.
    Recursively add image files to the list if the file is a directory.

    Parameters
    ----------
    files : list
        List of files

    Returns
    -------
    list
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
            files.extend(glob.glob(os.path.join(file, "*.jpeg")))
            files.extend(glob.glob(os.path.join(file, "*.jpg")))
            files.extend(glob.glob(os.path.join(file, "*.png")))
            # Also from subdirectories
            files.extend(glob.glob(os.path.join(file, "**/*.jpeg")))
            files.extend(glob.glob(os.path.join(file, "**/*.jpg")))
            files.extend(glob.glob(os.path.join(file, "**/*.png")))
            continue
        if not os.path.isfile(file):
            logging.warning("Cannot read %s, skipping ...", file)
            continue
        filenames.append(file)
    return filenames


# TODO Oops, this is not used yet
def process_images_in_process_pool(
    images_to_check: list, known_names: list, known_face_encodings: list
):
    """Process multiple images in a process pool.

    Parameters
    ----------
    images_to_check : list
        List of images to check
    known_names : list
        List of known names
    known_face_encodings : list
        List of known face encodings
    """
    # MacOS will crash due to a bug in libdispatch if you don’t use “forkserver”
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=None)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
    )

    pool.starmap(analyze_file, function_parameters)


def analyze_file(
    file: str, known_names: list, known_face_encodings: list, tolerance: float = 0.55
) -> list:
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
    list
        The list of recognized people
    """
    unknown_image = face_recognition.load_image_file(file)
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


def scan_known_people(known_people_folder: str) -> Tuple[list, list]:
    """Scan a folder with known people and return their name and encoding.

    Parameters
    ----------
    known_people_folder : str
        The name of the folder with known people

    Returns
    -------
    Tuple[list, list]
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
        try:
            idx = known_names.index(basename)
            if idx != -1 and filesize_bytes == known_filesize_bytes[idx]:
                logging.debug("Using cached encodings for %s.", file)
                continue
        except ValueError:
            # New entry for cache
            pass
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


def add_metadata(file: str, recognized_people: list):
    """Add list of recognized people to a file.

    Parameters
    ----------
    file : str
        File to add the metadata to
    recognized_people : list
        List of recognized people
    """
    # Found at least one recognized person
    current_metadata = (
        (
            subprocess.run(
                ["exiftool", "-Keywords", file], capture_output=True, check=True
            )
            .stdout.decode(encoding="ansi")
            .strip()
        )
        .encode("ansi")
        .decode("utf8")
    )
    if not "Personen" in current_metadata:
        # No person found before (assuming Bridge metadata format)
        exiftool_write(file, "Keywords", "Personen")
        exiftool_write(file, "HierarchicalSubject", "Personen")
        exiftool_write(file, "Subject", "Personen")

    for person in recognized_people:
        if person not in current_metadata:
            logging.debug("Found %s in this image. Updating metadata!", person)
            # Newly found person
            exiftool_write(file, "Keywords", person)
            exiftool_write(file, "HierarchicalSubject", f"Personen|{person}")
            exiftool_write(file, "Subject", person)
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
            "Referenzen Gesichtserkennung",
        ),
    )
    logging.debug("Using a tolerance of %f for face recognition.", tolerance)
    for idx, file in enumerate(scan_image_files(files), start=1):
        logging.info("Starting analysis of %s (%d/%d) ...", file, idx, len(files) - 1)
        recognized_people = analyze_file(
            file, known_names, known_face_encodings, tolerance
        )
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
                    "%d unknown person(s) detected. Please check manually!",
                    recognized_people.count("unknown_person"),
                )
                while recognized_people.count("unknown_person"):  # Might be multiple
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
                "No person—not even someone unknown—identified in this image."
            )


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    main(args["PATH"], tolerance=float(args["--tolerance"]))
