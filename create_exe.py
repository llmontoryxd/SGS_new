import PyInstaller.__main__
import shutil
import os

PyInstaller.__main__.run([
    'run.py',
    '--onefile'
])

shutil.copy('config.txt', 'dist/config.txt')