# Face Tagger
Tags recognized faces in images in Adobe Bridge compatible format.

Currently only tested and working on Windows 10.

## Usage
```
Usage: face_tagger PATH ...
"""
```

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
