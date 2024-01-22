# Face Tagger
Tags recognized faces in images in Adobe Bridge compatible format.

Currently only tested and working on Windows 10.

## Usage
```
Usage:
gesichtserkennung.py [-v] --references=<PATH> [--tolerance=<VALUE>] <PATH>...

-r PATH, --references <PATH>   Path to the folder with reference images
-t VALUE, --tolerance <VALUE>  Tolerance for face detection (higher: more false positives, less false negatives) [default: 0.55].
-v, --verbose                  Verbose output
```

Example:
```
C:\>exiftool photo_2022-05-22_05-24-38.jpg
ExifTool Version Number         : 12.44
File Name                       : photo_2022-05-22_05-24-38.jpg
Directory                       : c:/
File Size                       : 133 kB
File Modification Date/Time     : 2022:08:11 00:07:25+02:00
File Access Date/Time           : 2022:08:11 00:07:25+02:00
File Creation Date/Time         : 2022:08:10 23:57:59+02:00
File Permissions                : -rw-rw-rw-
File Type                       : JPEG
File Type Extension             : jpg
MIME Type                       : image/jpeg
Image Width                     : 640
Image Height                    : 640
Encoding Process                : Baseline DCT, Huffman coding
Bits Per Sample                 : 8
Color Components                : 3
Y Cb Cr Sub Sampling            : YCbCr4:2:0 (2 2)
Image Size                      : 640x640
Megapixels                      : 0.410

C:\>face_tagger -r references photo_2022-05-22_05-24-38.jpg
Starting analysis of photo_2022-05-22_05-24-38.jpg ...
Analysis of photo_2022-05-22_05-24-38.jpg finished. Proceeding adding recognized persons to the metadata.
Found <REDACTED> in this image. Updating metadata!
<REDACTED2> is already tagged in this image. Not adding them again.
3 unknown person(s) detected. Please check manually!

C:\>exiftool photo_2022-05-22_05-24-38.jpg
ExifTool Version Number         : 12.44
File Name                       : photo_2022-05-22_05-24-38.jpg
Directory                       : c:/
File Size                       : 137 kB
File Modification Date/Time     : 2022:08:11 00:05:46+02:00
File Access Date/Time           : 2022:08:11 00:05:46+02:00
File Creation Date/Time         : 2022:08:10 23:57:59+02:00
File Permissions                : -rw-rw-rw-
File Type                       : JPEG
File Type Extension             : jpg
MIME Type                       : image/jpeg
Current IPTC Digest             : c934de6c2ab7fa777532fc8a9307233a
Keywords                        : <REDACTED>, <REDACTED2>
Application Record Version      : 4
XMP Toolkit                     : Image::ExifTool 12.44
Subject                         : <REDACTED>, <REDACTED2>
Hierarchical Subject            : <REDACTED>, <REDACTED2>
Image Width                     : 640
Image Height                    : 640
Encoding Process                : Baseline DCT, Huffman coding
Bits Per Sample                 : 8
Color Components                : 3
Y Cb Cr Sub Sampling            : YCbCr4:2:0 (2 2)
Image Size                      : 640x640
Megapixels                      : 0.410
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

# Changelog
## 0.1.1
- Small bugfix in debug logging, code style improvements (breaking up long lines)

## 0.1.0
- Now processes video files (mp4, avi, mkv, mov) with `cv2` (analyzes the first video frame only)
