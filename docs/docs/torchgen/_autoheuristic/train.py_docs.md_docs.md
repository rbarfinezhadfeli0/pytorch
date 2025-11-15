# Documentation: `docs/torchgen/_autoheuristic/train.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/_autoheuristic/train.py_docs.md`
- **Size**: 8,672 bytes (8.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/_autoheuristic/train.py`

## File Metadata

- **Path**: `torchgen/_autoheuristic/train.py`
- **Size**: 5,911 bytes (5.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors

import argparse
import json
import warnings

import pandas as pd  # type: ignore[import-untyped]

from torch._inductor.autoheuristic.autoheuristic_utils import (
    CHOICE_COL,
    get_metadata_str_from_log,
)


# TODO (AlnisM): Fix these warnings
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.",
)


class AHTrain:
    """
    Base class for AutoHeuristic training.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self):
        self.parser.add_argument(
            "dataset",
            type=str,
            help="Path to text file containing data collected with AutoHeuristic.",
        )
        self.parser.add_argument(
            "--nrows",
            type=int,
            default=None,
            help="Only read first n rows of the dataset.",
        )
        self.parser.add_argument(
            "--heuristic-name",
            type=str,
            default="learned_heuristic",
            help="Name of the heuristic to be generated.",
        )
        self.parser.add_argument(
            "--data",
            nargs=2,
            action="append",
            metavar=("TYPE", "PATH"),
            help="Specify name of datasets and file paths to be evaluated.",
        )
        self.parser.add_argument(
            "--save-dot",
            action="store_true",
            help="Export heuristic to graphviz dot.",
        )
        self.parser.add_argument(
            "--ranking",
            type=int,
            default=None,
            help="""
                Makes AutoHeuristic learn a heuristic that ranks choices instead of predicting a single choice.
                The argument is the number of choices the heuristic will provide.
            """,
        )

    def parse_args(self):
        return self.parser.parse_args()

    def parse_log(self, log_path, nrows=None):
        (df, metadata) = self.deserialize_data(log_path)
        numerical_features = metadata["numerical_features"]
        categorical_features = metadata["categorical_features"]
        choices = df[CHOICE_COL].unique().tolist()
        features = numerical_features + categorical_features
        if nrows is not None:
            df = df.head(nrows)
        df = self.filter_df(df)
        return (df, metadata, features, categorical_features, choices)

    def generate_heuristic(self):
        self.args = self.parse_args()
        self.main(
            self.args.dataset,
            self.args.data,
            self.args.nrows,
            self.args.heuristic_name,
            self.args.save_dot,
            self.args.ranking is not None,
        )

    def filter_df(self, df):
        return df

    def add_new_features(self, results):
        return (results, [])

    def add_real_datasets(self, datasets, other_datasets, cat_feature2cats):
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path, cat_feature2cats=cat_feature2cats, apply_filters=False
                )
                datasets[name] = df_other

    def handle_categorical_features(
        self, cat_feature2cats, categorical_features, results
    ):
        # Doing this here because if we create another df for testing purposes
        # and that other df does not contain all categories for a categorical feature,
        # pd.dummies will not create columns for the missing categories
        if not cat_feature2cats:
            cat_feature2cats = {}
        for cat_feature in categorical_features:
            if cat_feature in cat_feature2cats:
                categories = cat_feature2cats[cat_feature]
            else:
                categories = results[cat_feature].unique()
                cat_feature2cats[cat_feature] = categories
            results[cat_feature] = pd.Categorical(
                results[cat_feature], categories=categories
            )

        dummy_col_2_col_val = {}
        for col in categorical_features:
            unique_vals = results[col].unique()
            for val in unique_vals:
                dummy_col_2_col_val[f"{col}_{val}"] = (col, val)
        # one-hot encode categorical features
        results = pd.get_dummies(results, columns=categorical_features)
        return (results, cat_feature2cats, dummy_col_2_col_val)

    def gen_precondition(self, opt_name, shared_memory, device_capa):
        return f"""    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == {shared_memory}
            and str(metadata.device_capa) == "{device_capa}"
        )"""

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, dt
    ):
        pass

    def gen_predict_fn_def(self):
        pass

    def write_heuristic_to_file(self, lines, heuristic_name):
        output_file = (
            f"../../../torch/_inductor/autoheuristic/artifacts/_{heuristic_name}.py"
        )
        path = f"{output_file}"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def deserialize_data(self, log_path):
        json_string = get_metadata_str_from_log(log_path)
        metadata = self.deserialize_metadata(json_string)

        df = pd.read_csv(log_path, skiprows=1, on_bad_lines="skip")
        return (df, metadata)

    def deserialize_metadata(self, json_string):
        return json.loads(json_string)


if __name__ == "__main__":
    train = AHTrain()
    train.generate_heuristic()

```



## High-Level Overview

"""    Base class for AutoHeuristic training.

This Python file contains 2 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AHTrain`

**Functions defined**: `__init__`, `add_base_arguments`, `parse_args`, `parse_log`, `generate_heuristic`, `filter_df`, `add_new_features`, `add_real_datasets`, `handle_categorical_features`, `gen_precondition`, `check_precondition`, `codegen_boilerplate`, `gen_predict_fn_def`, `write_heuristic_to_file`, `deserialize_data`, `deserialize_metadata`

**Key imports**: argparse, json, warnings, pandas as pd  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/_autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `json`
- `warnings`
- `pandas as pd  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torchgen/_autoheuristic`):

- [`train_regression.py_docs.md`](./train_regression.py_docs.md)
- [`merge_data.py_docs.md`](./merge_data.py_docs.md)
- [`generate_heuristic.sh_docs.md`](./generate_heuristic.sh_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`ah_tree.py_docs.md`](./ah_tree.py_docs.md)
- [`benchmark_runner.py_docs.md`](./benchmark_runner.py_docs.md)
- [`benchmark_utils.py_docs.md`](./benchmark_utils.py_docs.md)
- [`train_decision.py_docs.md`](./train_decision.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test.sh_docs.md`](./test.sh_docs.md)


## Cross-References

- **File Documentation**: `train.py_docs.md`
- **Keyword Index**: `train.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/_autoheuristic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/_autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torchgen/_autoheuristic`):

- [`ah_tree.py_kw.md_docs.md`](./ah_tree.py_kw.md_docs.md)
- [`test.sh_kw.md_docs.md`](./test.sh_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`train_regression.py_docs.md_docs.md`](./train_regression.py_docs.md_docs.md)
- [`collect_data.sh_docs.md_docs.md`](./collect_data.sh_docs.md_docs.md)
- [`benchmark_utils.py_docs.md_docs.md`](./benchmark_utils.py_docs.md_docs.md)
- [`benchmark_runner.py_kw.md_docs.md`](./benchmark_runner.py_kw.md_docs.md)
- [`requirements.txt_docs.md_docs.md`](./requirements.txt_docs.md_docs.md)
- [`test_utils.py_kw.md_docs.md`](./test_utils.py_kw.md_docs.md)
- [`requirements.txt_kw.md_docs.md`](./requirements.txt_kw.md_docs.md)


## Cross-References

- **File Documentation**: `train.py_docs.md_docs.md`
- **Keyword Index**: `train.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
