import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face_tagger",
    version="0.1.2",
    author="Jonas A. Wendorf",
    description="Tags recognized faces in images in Adobe Bridge compatible format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonasw234/face_tagger",
    packages=setuptools.find_packages(),
    install_requires=["face_recognition"],
    include_package_data=True,
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "OSI Approved :: GNU General Public License v3 or later (GPLv3)",
        "Operating System :: Windows",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": ["face_tagger=face_tagger.face_tagger:main"],
    },
)
