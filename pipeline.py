"""
pipeline.py
-----------
This file is responsible for pulling, processing, and storing the training data.

Usage:
    poetry run python pipeline.py
"""

import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional

GIT_URL = "https://github.com/DaveGamble/cJSON.git"


def clone_repo(git_url: str, output_dir: str):
    """
    Clone the repository from the given URL and save it to the output directory.
    """
    subprocess.run(["git", "clone", git_url, output_dir])


def compile_c_to_ll(file_path: Path) -> Optional[Path]:
    """
    Compile the given C file to LLVM IR and return the path to the output file.
    Retries with -I. if compilation fails the first time.
    Logs to compile_errors.log on failure.
    """
    output_path = file_path.with_suffix(".ll")
    cmd = ["clang", "-S", "-emit-llvm", "-O0", str(file_path), "-o", str(output_path)]

    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return output_path
    except subprocess.CalledProcessError:
        cmd_with_include = cmd[:-2] + ["-I."] + cmd[-2:]
        try:
            subprocess.run(
                cmd_with_include,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return output_path
        except subprocess.CalledProcessError:
            with open("compile_errors.log", "a") as log_file:
                log_file.write(f"Failed to compile {file_path} with -I.\n")
            return None


def rec_compile(dir_path: Path) -> List[Path]:
    """
    Recursively compile all C files in the given directory to LLVM IR using Path.rglob.
    Logs errors to compile_errors.log.
    """
    compiled_files = []
    for c_file in dir_path.rglob("*.c"):
        result = compile_c_to_ll(c_file)
        if result:
            print(f"Compiled: {result}")
            compiled_files.append(result)
        else:
            print(f"Failed: {c_file}")
    return compiled_files


def move_files(files: List[Path], output_dir: Path):
    """
    Move the C files and their compiled LLVM IR outputs to the data directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        # first move the C file
        # swap the .ll for .c in the name
        c_file = file.with_suffix(".c")
        shutil.move(c_file, output_dir / "c" / c_file.name)
        # move the .ll file
        shutil.move(file, output_dir / "ll" / file.name)


def main():
    """
    Main function that orchestrates the pipeline.
    """
    # if the data directory exists, delete it
    if Path("data").exists():
        shutil.rmtree(Path("data"))

    # if the compile_errors.log file exists, delete it
    if Path("compile_errors.log").exists():
        os.remove(Path("compile_errors.log"))

    # create the data directory
    Path("data").mkdir(parents=True, exist_ok=True)
    # create the c and ll directories
    Path("data/c").mkdir(parents=True, exist_ok=True)
    Path("data/ll").mkdir(parents=True, exist_ok=True)

    clone_repo(GIT_URL, "cJSON")
    ll_files = rec_compile(Path("cJSON"))
    move_files(ll_files, Path("data"))
    # clean up the cJSON directory
    shutil.rmtree(Path("cJSON"))


if __name__ == "__main__":
    main()
