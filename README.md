# motor-imagery-bci

Setup Steps:

1) Make sure you have HDF5 installed: [Download Link](https://www.hdfgroup.org/downloads/hdf5/)
2) Make sure you have python 3.8 installed [Download Link](https://www.python.org/downloads/release/python-380/) (choose "Executable installer")
3) Open Pycharm, choose python 3.8 as interpreter, and install all requirements (as prompted by dialog box)

Recording Instructions:
1) Make a copy of `recording_params.default.json` and name it `recording_params.json` and configure it as you whish (this file will not be tracked in git)
2) Visit `workflows.py` and run the function `run_session`
