# inline-ml
Predicting function inlining using LLVM IR + XGBoost

## Setup and Usage
This project uses [Poetry](https://python-poetry.org/) to manage dependencies. The project also externally depends on Clang and its toolchain. To install the dependencies, run:

```bash
poetry install
```

## Usage
To run the pipeline in order to pull and process the data, run:

```bash
poetry run python pipeline.py
```

In order to train the model, run:

```bash
poetry run python train_model.py
```

After training, the model will be saved to `inline_model.json`. To run the GUI with the model, run:

```bash
poetry run python dashboard.py
```

## Data Attribution

This project extracts data from a number of open-source C projects listed below:

- [cJSON](https://github.com/DaveGamble/cJSON) by [Dave Gamble](https://github.com/DaveGamble)
- [zlib](https://github.com/madler/zlib) by [Mark Adler](https://github.com/madler)
- [micropython](https://github.com/micropython/micropython) by [The MicroPython Team](https://github.com/micropython)
- [cosmopolitan](https://github.com/jart/cosmopolitan) by [Justine Tunney](https://github.com/jart)
- [blink](https://github.com/jart/blink) by [Justine Tunney](https://github.com/jart)
- [sectorlisp](https://github.com/jart/sectorlisp) by [Justine Tunney](https://github.com/jart)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by [ggml-org](https://github.com/ggml-org)
- [llama2.c](https://github.com/karpathy/llama2.c) by [Andrej Karpathy](https://github.com/karpathy)
- [blurhash](https://github.com/woltapp/blurhash) by [Wolt](https://github.com/woltapp)


I process the C code into LLVM IR using Clang. Then I use [llvmcpy](https://github.com/revng/llvmcpy) and some rough heuristics to extract the needed features from the IR that the model will train on.

## License 

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.