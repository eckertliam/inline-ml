"""
pipeline.py
-----------
This file is responsible for pulling, processing, and storing the training data.
Some of the functions are also used in dashboard.py to compile the user's C code to LLVM IR and extract the callsites.

Usage:
    poetry run python pipeline.py
"""

import os
from pathlib import Path
import shutil
import subprocess
import threading
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

def check_llvm_tools():
    for tool in ["opt", "clang"]:
        if not shutil.which(tool):
            raise RuntimeError(
                f"Required LLVM tool {tool} not found in PATH.\n"
                f"This tool is required to run this program.\n"
                f"Please follow the user guide to install the required tools."
            )


def read_fail_cache(path: Path) -> Set[Path]:
    # read the fail cache from a file
    with open(path, "r") as f:
        return set(Path(line.strip()) for line in f.readlines())


def write_fail_cache(path: Path, fail_cache: Set[Path]):
    # write the fail cache to a file
    with open(path, "w") as f:
        for fail in fail_cache:
            f.write(str(fail) + "\n")


def clone_repo(url: str) -> Optional[Path]:
    # set the output to the name of the repo
    output_dir = Path(url).stem
    # if the directory exists, delete it
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    # clone the repo
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, output_dir],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # check if the repo was cloned successfully
    if result.returncode != 0:
        # log the error to the terminal
        print(f"Failed to clone {url}: {result.stderr.decode(errors='replace')}")
        # return None
        return None
    else:
        # return the path to the repo
        return Path(output_dir)


def clone_repos(git_urls: List[str]) -> List[Path]:
    # lists paths to the cloned repos
    output_dirs = []
    # clone the repos concurrently
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(clone_repo, url) for url in git_urls]
        # wait for the repos to be cloned
        for future in as_completed(futures):
            result = future.result()
            # if the repo was cloned successfully, add the path to the list
            if result:
                output_dirs.append(Path(result))

    return output_dirs


def compile_c_to_ll(
    file_path: Path, fail_cache: Optional[Set[Path]] = None, fail_cache_lock: Optional[threading.Lock] = None
) -> Optional[str]:
    # compile c to llvm ir using clang
    cmd = ["clang", "-S", "-emit-llvm", "-O2", "-I.", str(file_path), "-o", "-"]
    # try to run the command
    try:
        # run the command and capture the output
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # return the llvm ir
        return result.stdout.decode(errors="replace")
    except subprocess.CalledProcessError as e:
        # log the error to the compile_errors.log file
        with open("compile_errors.log", "a") as f:
            f.write(f"Failed to compile {file_path}\n")
            # log the error message
            f.write(e.stderr.decode(errors="replace") + "\n")
        if fail_cache is not None and fail_cache_lock is not None:
            # add the file to the fail cache with a lock
            with fail_cache_lock:
                fail_cache.add(file_path)
        return None


def rec_compile(dirs: List[Path], fail_cache: Optional[Set[Path]] = None) -> List[str]:
    # list to store the llvm ir
    ir_strings = []

    # flatten the list of files to compile
    c_files = [c_file for dir in dirs for c_file in dir.rglob("*.c")]

    # if the fail cache exists, filter out files that are in the fail cache
    if fail_cache is not None:
        c_files = [c_file for c_file in c_files if c_file not in fail_cache]
        fail_cache_lock = threading.Lock()
    else:
        fail_cache_lock = None

    # compile the files concurrently
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(compile_c_to_ll, c_file, fail_cache, fail_cache_lock) for c_file in c_files
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:  # add successful runs to the list
                ir_strings.append(result)

    return ir_strings


def main(keep_compiler_errors: bool = False):
    # check if the required LLVM tools are in the PATH
    check_llvm_tools()

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

    print("Cloning repositories...")
    dirs: List[Path] = clone_repos(GIT_URLS)
    # flush the terminal
    print("Compiling C files to LLVM IR...")
    ir_strings: List[str] = rec_compile(dirs, fail_cache)
    # write the fail cache to a file
    write_fail_cache(Path("fail_cache.txt"), fail_cache)
    # delete the cloned repos
    print("Cleaning up...")
    for dir in dirs:
        shutil.rmtree(dir)

    if not keep_compiler_errors and Path("compile_errors.log").exists():
        os.remove(Path("compile_errors.log"))

    # convert the ll files into a list of dataframes and concatenate them
    print("Collecting callsites from LLVM IR...")
    df = pd.concat([mod2df(ir_string) for ir_string in ir_strings], ignore_index=True)

    # print the number of callsites extracted
    print(f"Extracted {len(df)} callsites")

    # print the ratio of inlining decisions
    print(f"Ratio of inlining decisions: {df['llvm_inlining_decision'].mean()}")

    # write the dataframe to a csv file
    df.to_csv(Path("data/data.csv"), index=False)


if __name__ == "__main__":
    main()
