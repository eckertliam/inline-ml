"""
ir2vec.py
-----------
This file is responsible for extracting feature vectors from the LLVM IR.
The following features are extracted:
- Callee Function Name (string) - the name of the function being called
- Caller Function Name (string) - the function which contains the call
- Callee Function Size (int): the number of instructions in the callee function
- Caller Function Size (int): the number of instructions in the caller function
- Callee Callsites (int): the number of times the callee function is called cross module
- Inline Ratio (float): callee size / caller size
- Callee basic blocks (int): the number of basic blocks in the callee function
- Caller basic blocks (int): the number of basic blocks in the caller function
- Callee is marked inline (bool): whether the callee function is marked inline
- Callee is recursive (bool): whether the caller function is recursive
- Callee arg count (int): the number of arguments passed to the callee function
- Callee load store ratio (float): the ratio of loads to stores in the callee function
- LLVM inlining decision (bool): whether the LLVM optimizer decided to inline the function

Usage:
    poetry run python ir2vec.py
"""

from pathlib import Path
from typing import Dict, List, Optional
from llvmcpy import LLVMCPy

# TODO: integrate llvm opt inlining decisions into function features and inlining feature vectors

llvm = LLVMCPy()

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
        marked_inline (bool): Whether the function is marked inline
        arg_count (int): The number of arguments passed to the function
        loads (int): The number of loads in the function
        stores (int): The number of stores in the function
        load_store_ratio (float): The ratio of loads to stores in the function
    """

    def __init__(
        self,
        function_name: str
    ):
        self.function_name = function_name
        self.instruction_count: int = 0
        self.total_calls: int = 0
        self.calls_in_function: list[str] = []
        self.basic_blocks: int = 0
        self.is_recursive: bool = False
        self.marked_inline: bool = False
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
        callee_marked_inline (bool): Whether the callee is explicitly marked with the 'alwaysinline' attribute
        callee_is_recursive (bool): Whether the callee makes a recursive call to itself
        callee_arg_count (int): Number of arguments the callee function takes
        callee_load_store_ratio (float): Ratio of load to store instructions in the callee
        llvm_inlining_decision (bool): Whether the LLVM optimizer decided to inline the function
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
            else float('inf')
        )
        self.callee_basic_blocks = callee.basic_blocks
        self.caller_basic_blocks = caller.basic_blocks
        self.callee_marked_inline = callee.marked_inline
        self.callee_is_recursive = callee.is_recursive
        self.callee_arg_count = callee.arg_count
        self.callee_load_store_ratio = callee.load_store_ratio
        self.llvm_inlining_decision = False # TODO: figure out how to get the LLVM inlining decision

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
            "callee_marked_inline": self.callee_marked_inline,
            "callee_is_recursive": self.callee_is_recursive,
            "callee_arg_count": self.callee_arg_count,
            "callee_load_store_ratio": self.callee_load_store_ratio,
        }

def analyze_instruction(instr, current_fn_features: FunctionFeatures, features: Dict[str, FunctionFeatures]) -> None:
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
                analyze_instruction(instr, function_features, features)
                

        function_features.arg_count = 0 # TODO: figure out how to get the arg count
        function_features.marked_inline = False # TODO: figure out how to get the marked inline attribute

    return features


def extract_feature_vectors(func_features: Dict[str, FunctionFeatures]) -> List[InliningFeatureVector]:
    for fn in func_features.values():
        print(fn.function_name, fn.total_calls)
    vectors = []
    for caller in func_features.values():
        for callee_name in set(caller.calls_in_function):
            if callee_name in func_features:
                callee = func_features[callee_name]
                vector = InliningFeatureVector(callee=callee, caller=caller)
                vectors.append(vector)
    return vectors


def mod2vec(module_path: Path) -> List[InliningFeatureVector]:
    buffer = llvm.create_memory_buffer_with_contents_of_file(str(module_path))  # type: ignore
    context = llvm.get_global_context()  # type: ignore
    module = context.parse_ir(buffer)  # type: ignore

    func_features = extract_function_features(module)
    vectors = extract_feature_vectors(func_features)
    return vectors

