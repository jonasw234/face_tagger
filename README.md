# Face Tagger
Tags recognized faces in images in Adobe Bridge compatible format.

Currently only tested and working on Windows 10.

## Usage
```
Usage: face_tagger PATH ...
```

Example:
```
C:\>face_tagger photo_2022-05-22_05-24-38.jpg
Starting analysis of photo_2022-05-22_05-24-38.jpg ...
Analysis of photo_2022-05-22_05-24-38.jpg finished. Proceeding adding recognized persons to the metadata.
Found <REDACTED> in this image. Updating metadata!
<REDACTED> is already tagged in this image. Not adding them again.
3 unknown person(s) detected. Please check manually!
```

Warning, this can take a while depending on the performance of your system. This script automatically uses all the available CPUs to speed it up as much as possible, but it still takes ~ 30 seconds on my computer to process a single image.

## Installation
The [dlib](http://dlib.net/) dependency needs [CMake](https://cmake.org/) to install.  
Furthermore [exiftool](https://exiftool.org/) needs to be available in the PATH at runtime.  
For the development version:
```
git clone https://github.com/jonasw234/face_tagger
cd face_tagger
python3 setup.py install
pip3 install -r dev-requirements.txt
```
For normal usage do the same but don’t include the last line or use [`pipx`](https://pypi.org/project/pipx/) and install with
```
pipx install git+https://github.com/jonasw234/face_tagger
```
