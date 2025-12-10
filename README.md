Updated editor - README

1. Overview
-----------
This application (updatededitor.py) is a GUI tool for viewing signals, annotating events, and saving them in a legacy-compatible CSV format.

You can run it directly with Python or create an executable so you don’t have to open it from a terminal each time.

2. Requirements
---------------
- Python 3.9 or newer (installed system-wide or in a virtual environment)
- Required packages:
    dash
    plotly
    pyabf
    pandas
    numpy
    scipy

3. Installing Dependencies
--------------------------
Open a terminal (Command Prompt/PowerShell on Windows, Terminal on macOS) and run:

Windows:
    pip install dash plotly pyabf pandas numpy scipy

macOS:
    pip3 install dash plotly pyabf pandas numpy scipy

4. Running the Script
---------------------
Windows:
    1. Open Command Prompt
    2. Navigate to the folder containingupdatededitor.py
       Example:
           cd C:\Users\YourName\Downloads\updatededitor
    3. Run:
           python updatededitor.py

macOS:
    1. Open Terminal
    2. Navigate to the folder containing updatededitor.py
       Example:
           cd /Users/YourName/Downloads/updatededitor
    3. Run:
           python3 updatededitor.py

When prompted for channel names, select: 0, 2, 3
The event file selection is optional.

5. Saving Events
----------------
When you finish annotating, click "Save Reviewed Events".
The file will be saved in the same folder as your ABF file with "_event.csv" appended to the filename.

6. Creating an Executable
-------------------------
This will let you run QC Editor without needing to open a terminal.

Windows:
    1. Install PyInstaller:
           pip install pyinstaller
    2. In the folder containing updatededitor.py, run:
           pyinstaller --onefile updatededitor.py
    3. The executable will appear in the "dist" folder as updatededitor.exe

macOS:
    1. Install PyInstaller:
           pip3 install pyinstaller
    2. In the folder containing updatededitor.py, run:
           pyinstaller --onefileupdatededitor.py
    3. The application will appear in the "dist" folder as QC_editor.app

Note: The macOS .app build is untested. Please report compatibility.

7. Troubleshooting
------------------
- If you see "pip not recognized", use:
      python -m pip install <package>
  or on macOS:
      python3 -m pip install <package>

- If the browser doesn’t open automatically, check the terminal output for the local URL (usually http://127.0.0.1:8050) and open it manually.

8. Contact
----------
If you encounter issues, please contact Subrat Bastola.
