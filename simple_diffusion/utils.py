import os
import sys
import time

import requests

if sys.platform == 'darwin':
    from AppKit import NSURL, NSWorkspace


class ChangeDirectory:
    def __init__(self, dir) -> None:
        self.dir = dir
        self.orig_dir = os.getcwd()

    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.orig_dir)

class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        if self.name:
            print(f"{self.name} took {elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {elapsed_time:.6f} seconds")

def resource_path(relative_path):
    return os.path.join('simple_diffusion/resources', relative_path)

def reveal_in_finder(file_path):
    if sys.platform == 'darwin':
        file_url = NSURL.fileURLWithPath_(file_path)
        NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([file_url])

def download_file(url: str, output_path: str) -> None:
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f'Failed to download the file, status code: {response.status_code}')
