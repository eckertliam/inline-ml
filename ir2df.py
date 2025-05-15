"""
ir2df.py
-----------
This file is responsible for extracting feature vectors from the LLVM IR and converting them into a pandas DataFrame.
The following features are extracted:
- Callee Function Name (string) - the name of the function being called
- Caller Function Name (string) - the function which contains the call
- Callee Function Size (int): the number of instructions in the callee function
- Caller Function Size (int): the number of instructions in the caller function
- Callee Callsites (int): the number of times the callee function is called cross module
- Inline Ratio (float): callee size / caller size
- Callee basic blocks (int): the number of basic blocks in the callee function
- Caller basic blocks (int): the number of basic blocks in the caller function
- Callee is recursive (bool): whether the caller function is recursive
- Callee arg count (int): the number of arguments passed to the callee function
- Callee load store ratio (float): the ratio of loads to stores in the callee function
- LLVM inlining decision (bool): whether the LLVM optimizer decided to inline the callee function

Usage:
    poetry run python ir2df.py
"""

from pathlib import Path
import subprocess
import tempfile
import pandas as pd
from typing import Dict, Tuple
from llvmcpy import LLVMCPy
import yaml

llvm = LLVMCPy()


# we need this to ignore the odd yaml format that llvm opt produces
class IgnoreUnknownTagsLoader(yaml.SafeLoader):
    pass


def ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


IgnoreUnknownTagsLoader.add_multi_constructor("!", ignore_unknown)


class FunctionFeatures:
    """
    Tracks features per function.

    Attributes:
        function_name (str): The name of the function
        instruction_count (int): The number of instructions in the function
        total_calls (int): The number of times the function is called cross module
        calls_in_function (list[str]): The names of the functions that are called within the function
        basic_blocks (int): The number of basic blocks in the function
        is_recursive (bool): Whether the function is recursive
        arg_count (int): The number of arguments passed to the function
        loads (int): The number of loads in the function
        stores (int): The number of stores in the function
        load_store_ratio (float): The ratio of loads to stores in the function
    """

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.instruction_count: int = 0
        self.total_calls: int = 0
        self.calls_in_function: list[str] = []
        self.basic_blocks: int = 0
        self.is_recursive: bool = False
        self.arg_count: int = 0
        self.loads: int = 0
        self.stores: int = 0

    @property
    def load_store_ratio(self) -> float:
        if self.stores == 0:
            return 0
        return self.loads / self.stores


class InliningFeatureVector:
    """
    Represents the combined inlining feature vector for a (caller, callee) pair.

    Attributes:
        callee_name (str): The name of the callee function
        caller_name (str): The name of the caller function
        callee_instruction_count (int): Number of LLVM IR instructions in the callee function
        caller_instruction_count (int): Number of LLVM IR instructions in the caller function
        callee_total_calls (int): Total number of times the callee is invoked across the module
        inline_ratio (float): Ratio of callee instruction count to caller instruction count
        callee_basic_blocks (int): Number of basic blocks in the callee function
        caller_basic_blocks (int): Number of basic blocks in the caller function
        callee_is_recursive (bool): Whether the callee makes a recursive call to itself
        callee_arg_count (int): Number of arguments the callee function takes
        callee_load_store_ratio (float): Ratio of load to store instructions in the callee
        llvm_inlining_decision (bool): Whether the LLVM optimizer decided to inline the callee function
    """

    def __init__(self, callee: FunctionFeatures, caller: FunctionFeatures):
        self.callee_name = callee.function_name
        self.caller_name = caller.function_name
        self.callee_instruction_count = callee.instruction_count
        self.caller_instruction_count = caller.instruction_count
        self.callee_total_calls = callee.total_calls
        self.inline_ratio = (
            callee.instruction_count / caller.instruction_count
            if caller.instruction_count != 0
            else float("inf")
        )
        self.callee_basic_blocks = callee.basic_blocks
        self.caller_basic_blocks = caller.basic_blocks
        self.callee_is_recursive = callee.is_recursive
        self.callee_arg_count = callee.arg_count
        self.callee_load_store_ratio = callee.load_store_ratio
        self.llvm_inlining_decision = False

    def to_dict(self):
        return {
            "callee_name": self.callee_name,
            "caller_name": self.caller_name,
            "callee_instruction_count": self.callee_instruction_count,
            "caller_instruction_count": self.caller_instruction_count,
            "callee_total_calls": self.callee_total_calls,
            "inline_ratio": self.inline_ratio,
            "callee_basic_blocks": self.callee_basic_blocks,
            "caller_basic_blocks": self.caller_basic_blocks,
            "callee_is_recursive": self.callee_is_recursive,
            "callee_arg_count": self.callee_arg_count,
            "callee_load_store_ratio": self.callee_load_store_ratio,
            "llvm_inlining_decision": self.llvm_inlining_decision,
        }


def analyze_instruction(
    instr, current_fn_features: FunctionFeatures, features: Dict[str, FunctionFeatures]
) -> None:
    # NOTE: for some reason llvmcpy opcodes do not match up to their actual opcodes so I just print the instruction and parse it manually
    instr_str = instr.print_value_to_string()
    instr_str = instr_str.decode("utf-8")
    if "call" in instr_str.lower():
        if "@" in instr_str:
            # trim everything before and including the first @
            split_at = instr_str.split("@")[1]
            # trim everything after and including the first (
            split_param = split_at.split("(")[0]
            # trim whitespace
            fn_name = split_param.strip()
            # add the fn_name to the calls_in_function list
            current_fn_features.calls_in_function.append(fn_name)
            # check if the called function is in the features dictionary
            if fn_name in features:
                features[fn_name].total_calls += 1
            elif fn_name == current_fn_features.function_name:
                current_fn_features.is_recursive = True
        else:
            # this means its a call to an external function
            # we need to count the number of function calls so we just add "external" to the calls_in_function list
            current_fn_features.calls_in_function.append("external")
    elif "load" in instr_str.lower():
        current_fn_features.loads += 1
    elif "store" in instr_str.lower():
        current_fn_features.stores += 1


def extract_function_features(module) -> Dict[str, FunctionFeatures]:
    features = {}
    # do a first pass to enter functions into the features dictionary
    for function in module.iter_functions():
        if function.is_declaration():  # filter out functions that are only declarations
            continue
        fn_name = function.name.decode("utf-8")
        features[fn_name] = FunctionFeatures(fn_name)

    # do a second pass to extract features
    for function in module.iter_functions():
        if function.is_declaration():
            continue

        function_features = features[function.name.decode("utf-8")]

        for bb in function.iter_basic_blocks():
            function_features.basic_blocks += 1
            for instr in bb.iter_instructions():
                function_features.instruction_count += 1
                analyze_instruction(instr, function_features, features)

        function_features.arg_count = function.count_params()

    return features


def extract_feature_vectors(
    func_features: Dict[str, FunctionFeatures],
) -> Dict[Tuple[str, str], InliningFeatureVector]:
    vectors: Dict[Tuple[str, str], InliningFeatureVector] = {}
    for caller in func_features.values():
        for callee_name in set(caller.calls_in_function):
            if callee_name in func_features:
                callee = func_features[callee_name]
                vector = InliningFeatureVector(callee=callee, caller=caller)
                vectors[(caller.function_name, callee_name)] = vector
    return vectors


def get_llvm_inlining_decision(module_path: Path) -> Dict[Tuple[str, str], bool]:
    # create a temporary file to store the output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
        output_yaml = Path(temp_file.name)

    cmd = [
        "opt",
        f"-passes=inline",
        f"-inline-threshold=10000",
        f"-pass-remarks=inline",
        f"-pass-remarks-output={str(output_yaml)}",
        "-disable-output",  # we don’t need the resulting IR
        str(module_path),
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"opt failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # Dict[Tuple[callee_name, caller_name], inlining_decision]
    inlining_decisions: Dict[Tuple[str, str], bool] = {}

    with open(output_yaml, "r") as f:
        docs = list(yaml.load_all(f, Loader=IgnoreUnknownTagsLoader))

    for entry in docs:
        # just a guard, don't expect this to throw
        assert isinstance(entry, dict), f"entry is not a dict: {entry}"

        # skip if the pass is not inline
        if entry.get("Pass") != "inline":
            continue

        # prepare the callee and caller names
        callee_name = None
        caller_name = None

        # iterate over the arguments and extract the callee and caller names
        for arg in entry.get("Args", []):
            if isinstance(arg, dict):
                if "Callee" in arg:
                    callee_name = arg["Callee"]
                if "Caller" in arg:
                    caller_name = arg["Caller"]

        # guard against missing callee or caller names
        assert callee_name is not None, f"callee name is None: {entry}"
        assert caller_name is not None, f"caller name is None: {entry}"

        # extract the inlining decision
        was_inlined = entry.get("Name") == "Inlined"
        inlining_decisions[(callee_name, caller_name)] = was_inlined

    # delete the output yaml file
    output_yaml.unlink()

    return inlining_decisions


def mod2df(module_str: str) -> pd.DataFrame:
    """
    Extracts feature vectors from a string of LLVM IR and returns a pandas DataFrame.

    Args:
        module_str (str): The LLVM IR module as a string

    Returns:
        pd.DataFrame: A pandas DataFrame containing the feature vectors
    """
    # create a temporary file to store the module
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ll") as temp_file:
        temp_file.write(module_str.encode("utf-8"))
        module_path = Path(temp_file.name)

    inlining_decisions = get_llvm_inlining_decision(module_path)

    buffer = llvm.create_memory_buffer_with_contents_of_file(str(module_path))  # type: ignore
    context = llvm.get_global_context()  # type: ignore
    module = context.parse_ir(buffer)  # type: ignore

    func_features = extract_function_features(module)
    vector_dict = extract_feature_vectors(func_features)

    # delete the temporary file
    module_path.unlink()

    # pass over the vector dict and add the inlining decisions and add inlining decisions to feature vectors
    for (caller_name, callee_name), vector in vector_dict.items():
        vector.llvm_inlining_decision = inlining_decisions[(callee_name, caller_name)]

    # convert the vector dict to a dataframe
    df = pd.DataFrame([vector.to_dict() for vector in vector_dict.values()])
    return df
