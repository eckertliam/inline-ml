# inline-ml
Predicting function inlining using LLVM IR + XGBoost

## Data Attribution

This project uses C code from [cJSON](https://github.com/DaveGamble/cJSON) by [Dave Gamble](https://github.com/DaveGamble) as a dataset.
I process the C code into LLVM IR using Clang. Then I use [llvmcpy](https://github.com/revng/llvmcpy) and some rough heuristics to extract the needed
features from the IR that we train the model on.

## License 

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.