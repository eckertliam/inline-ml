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
from typing import List, Optional, Set

import pandas as pd
from ir2df import mod2df
from concurrent.futures import ThreadPoolExecutor, as_completed

GIT_URLS = [
    "https://github.com/DaveGamble/cJSON.git",
    "https://github.com/madler/zlib.git",
    "https://github.com/micropython/micropython.git",
    "https://github.com/jart/cosmopolitan.git",
    "https://github.com/jart/blink.git",
    "https://github.com/jart/sectorlisp.git",
    "https://github.com/ggml-org/llama.cpp.git",
    "https://github.com/karpathy/llama2.c.git",
    "https://github.com/woltapp/blurhash.git",
]


def read_fail_cache(path: Path) -> Set[Path]:
    """
    Read the fail cache from the given path.
    """
    with open(path, "r") as f:
        return set(Path(line.strip()) for line in f.readlines())


def write_fail_cache(path: Path, fail_cache: Set[Path]):
    """
    Write the fail cache to the given path.
    """
    with open(path, "w") as f:
        for fail in fail_cache:
            f.write(str(fail) + "\n")


def clone_repo(url: str) -> Optional[Path]:
    output_dir = Path(url).stem
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    result = subprocess.run(
        ["git", "clone", url, output_dir],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Path(output_dir) if result.returncode == 0 else None


def clone_repos(git_urls: List[str]) -> List[Path]:
    """
    Clone the repositories from the given URLs concurrently using ThreadPoolExecutor.
    """
    output_dirs = []

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(clone_repo, url) for url in git_urls]
        for future in as_completed(futures):
            result = future.result()
            if result:
                output_dirs.append(Path(result))

    return output_dirs


def compile_c_to_ll(file_path: Path) -> Optional[Path]:
    """
    Compile the given C file to LLVM IR and return the path to the output file.
    Retries with -I. if compilation fails the first time.
    Logs to compile_errors.log on failure.
    """
    output_path = file_path.with_suffix(".ll")
    cmd = ["clang", "-S", "-emit-llvm", "-O2", str(file_path), "-o", str(output_path)]

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


def rec_compile(dirs: List[Path], fail_cache: Set[Path] = set()) -> List[Path]:
    """
    Recursively compile all C files in the given directories to LLVM IR using Path.rglob.
    Logs errors to compile_errors.log.
    """
    compiled_files = []

    # flatten the list of files to compile
    c_files = [c_file for dir in dirs for c_file in dir.rglob("*.c")]
    # filter out files that are in the fail cache
    c_files = [c_file for c_file in c_files if c_file not in fail_cache]

    # compile the files
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(compile_c_to_ll, c_file) for c_file in c_files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                compiled_files.append(result)

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
        if not c_file.exists():
            print(f"C file does not exist: {c_file}")
            continue
        shutil.move(c_file, output_dir / "c" / c_file.name)
        # move the .ll file
        shutil.move(file, output_dir / "ll" / file.name)


def main(keep_compiler_errors: bool = False):
    """
    Main function that orchestrates the pipeline.
    """
    if Path("fail_cache.txt").exists():
        fail_cache = read_fail_cache(Path("fail_cache.txt"))
    else:
        fail_cache = set()

    # if the data directory exists, delete it
    if Path("data").exists():
        shutil.rmtree(Path("data"))

    # if the compile_errors.log file exists, delete it
    if Path("compile_errors.log").exists():
        os.remove(Path("compile_errors.log"))

    # create the data directory
    Path("data").mkdir(parents=True, exist_ok=True)
    # create the c, ll, and csv directories
    Path("data/c").mkdir(parents=True, exist_ok=True)
    Path("data/ll").mkdir(parents=True, exist_ok=True)
    Path("data/csv").mkdir(parents=True, exist_ok=True)

    print("Cloning repositories...")
    dirs: List[Path] = clone_repos(GIT_URLS)
    # flush the terminal
    print("Compiling C files to LLVM IR...")
    ll_files = rec_compile(dirs, fail_cache)
    # write the fail cache to a file
    write_fail_cache(Path("fail_cache.txt"), fail_cache)
    print("Moving files...")
    move_files(ll_files, Path("data"))
    # delete the cloned repos
    print("Cleaning up...")
    for dir in dirs:
        shutil.rmtree(dir)

    if not keep_compiler_errors and Path("compile_errors.log").exists():
        os.remove(Path("compile_errors.log"))

    # convert the ll files into a list of dataframes and concatenate them
    print("Collecting callsites from LLVM IR...")
    df = pd.concat(
        [mod2df(ll_file) for ll_file in Path("data/ll").glob("*.ll")], ignore_index=True
    )

    # print the number of callsites extracted
    print(f"Extracted {len(df)} callsites")

    # write the dataframe to a csv file
    df.to_csv(Path("data/csv/data.csv"), index=False)


if __name__ == "__main__":
    main()
