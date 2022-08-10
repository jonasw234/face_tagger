#!/usr/bin/env python3
"""Tries to recognize faces in input image list and update file’s metadata.

Usage:
    face_tagger.py PATH ...
"""
import glob
import os
import subprocess
import sys


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


def main(files: list):
    """Read list of files, recognize faces and update metadata.

    Parameters
    ----------
    files : str
        List of input images
    """
    if not files:
        print("No files supplied to recognize!")
        return
    for file in files:
        if os.path.isdir(file):
            print(
                f"Adding *.jpeg, *.jpg and *.png files in {file} directory to input list."
            )
            # Add images in directory
            files.extend(glob.glob(os.path.join(file, "*.jpeg")))
            files.extend(glob.glob(os.path.join(file, "*.jpg")))
            files.extend(glob.glob(os.path.join(file, "*.png")))
            continue
        if not os.path.isfile(file):
            print(f"Cannot read {file}, skipping ...")
            continue
        print(f"Starting analysis of {file} ...")
        face_recognition = subprocess.run(
            [
                "face_recognition",
                "--cpus",
                "-1",
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "reference_images",
                ),
                file,
            ],
            capture_output=True,
            check=True,
        )
        recognized_persons = [
            person.split(",")[1]
            for person in face_recognition.stdout.decode(encoding="ansi")
            .strip()
            .split("\r\n")
        ]
        print(
            f"Analysis of {file} finished. Proceeding adding recognized persons to the metadata."
        )
        if len(recognized_persons) and not "no_persons_found" in recognized_persons:
            if recognized_persons.count("unknown_person"):
                print(
                    f"{recognized_persons.count('unknown_person')} unknown person(s) detected. "
                    "Please check manually!"
                )
                while recognized_persons.count("unknown_person"):  # Might be multiple
                    recognized_persons.remove("unknown_person")
            if not recognized_persons:
                # Only unknown persons were identified, so we can stop here
                continue
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
            for person in recognized_persons:
                if person not in current_metadata:
                    print(f"Found {person} in this image. Updating metadata!")
                    # Newly found person
                    exiftool_write(file, "Keywords", person)
                    exiftool_write(
                        file, "HierarchicalSubject", person
                    )  # Top|Middle|Bottom
                    exiftool_write(file, "Subject", person)
                else:
                    print(
                        f"{person} is already tagged in this image. Not adding them again."
                    )
        else:
            print("No persons – not even unknown ones – identified in this image.")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
