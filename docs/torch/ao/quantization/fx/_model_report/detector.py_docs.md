# Documentation: `torch/ao/quantization/fx/_model_report/detector.py`

## File Metadata

- **Path**: `torch/ao/quantization/fx/_model_report/detector.py`
- **Size**: 76,891 bytes (75.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.ao.nn.qat as nnqat
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx._equalize import (
    default_equalization_qconfig,
    EqualizationQConfig,
)
from torch.ao.quantization.fx._model_report.model_report_observer import (
    ModelReportObserver,
)
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    default_dynamic_quant_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_weight_observer,
    ObserverBase,
)
from torch.ao.quantization.qconfig import (
    _assert_valid_qconfig,
    default_qconfig,
    QConfig,
)


# Names for observer insert keys
DETECTOR_TARGET_NODE_KEY = "target_node"
DETECTOR_OBS_TO_INSERT_KEY = "observer_to_insert"
DETECTOR_IS_POST_OBS_KEY = "is_post_observer"
DETECTOR_OBS_ARGS_KEY = "observer_args"


# Mapping related code
class DetectorQConfigInfo:
    r"""
    This class contains the QConfig information for a single module.
    The list of variables / values this contains can grow depending on the
    extensibility of the qconfig mapping feature set but this currently includes:
    - if activation observer is dynamic
    - if weight observer is per channel


    Args:
        module_fqn (str): The fully qualified name (fqn) of the module that this
            information contains info relevant to qconfig for
    """

    def __init__(self, module_fqn: str):
        super().__init__()
        self.module_fqn = module_fqn

        # populate this section with all the variables we might find important
        # change from none if your detector is actually using this
        self.is_activation_dynamic = False
        self.is_weight_per_channel = False

        # equalization related options
        self.is_equalization_recommended = False

    def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
        r"""
        Args:
            module (torch.nn.Module) The module we are generating
            the qconfig for

        Returns the generated quantization QConfig according to what a valid configuration is
        """
        # Apply suggestions to new qconfig
        module_qconfig = default_qconfig

        # keep track of dynamic and per_channel recommendations
        recommendations_list = []
        # append as if a list of combinations
        recommendations_list.append(
            (self.is_activation_dynamic, self.is_weight_per_channel)
        )
        recommendations_list.append(
            (self.is_activation_dynamic, False)
        )  # only trying dynamic rec
        recommendations_list.append(
            (False, self.is_weight_per_channel)
        )  # only trying dynamic

        # now we try each of the combinations
        for rec in recommendations_list:
            # rec[0] -> dynamic recommended
            # rec[1] -> per channel recommended
            activation = default_dynamic_quant_observer if rec[0] else default_observer
            weight = (
                default_per_channel_weight_observer
                if rec[1]
                else default_weight_observer
            )
            test_config = QConfig(activation, weight)
            try:
                _assert_valid_qconfig(test_config, module)
                module_qconfig = test_config
                break
            except AssertionError:
                # if not a valid configuration, we move on to the next one in priority
                continue

        # return the QConfig chosen
        return module_qconfig

    def generate_equalization_qconfig(self) -> EqualizationQConfig:
        r"""
        This returns the equalization configuration for a module.

        For now, it just returns the default, but as more equalization options become
        possible, this method can get more fleshed out with more nuanced granularity.


        Returns the generated equalization QConfig according to what a valid configuration is
        """
        # in this case, we just return default equalization config
        # we know this is valid because only valid modules would even
        # have this option
        return default_equalization_qconfig


# Adding base class for detectors
class DetectorBase(ABC):
    r"""Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self) -> None:
        super().__init__()
        self.detector_config_info = None

    @abstractmethod
    def determine_observer_insert_points(self, model) -> dict:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.
            This dict maps string keys to detector specific information
        """

    @abstractmethod
    def get_detector_name(self) -> str:
        r"""Returns the name of the current detector"""

    @abstractmethod
    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """

    def _get_targeting_node(
        self, prepared_fx_model: GraphModule, target_fqn: str
    ) -> torch.fx.node.Node:
        r"""
        Takes in a GraphModule and the target_fqn and finds the node whose target is this fqn.

        If it's not found, it means it is most likely inside a fused layer
            We just go one layer up in terms of the fqn we are searching for until we find parent node
            If we get to empty string, then we know that it doesn't exist

        The reason for the recursion is that if the model that we are looking for got fused,
        we will have module fqn as e.g. x.linear.0 but the graph will only have a node for the fused module,
        which would have fqn as x.linear so they will not match.
        To handle this, if we don't match, we then take off the last bit of the fqn e.g. x.linear.0 -> x.linear,
        or more generally foo.bar.baz -> foo.bar and search again, this will allow us to locate the correct module
        even in cases with fusion

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        for node in prepared_fx_model.graph.nodes:
            # if the node's target is our target, return it
            if node.target == target_fqn:
                return node

        # getting here means node not found
        # if no "." we are already at base and failed
        parent_fqn_sep_index = target_fqn.rfind(".")
        if parent_fqn_sep_index == -1:
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            # recursively call it with parent fqn
            return self._get_targeting_node(
                prepared_fx_model, target_fqn[:parent_fqn_sep_index]
            )

    @abstractmethod
    def generate_detector_report(self, model) -> tuple[str, dict[str, Any]]:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Str: string report of the suggested improvements
            Dict: contains useful data collected by the observer pertinent to this report
        """


class PerChannelDetector(DetectorBase):
    r"""This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
    Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

    per_channel quantization can lead to major benefits in the form of accuracy.
    Therefore, if the backend used by the user supports it, it is recommended to use

    Args:
        backend (str, optional): the backend the user wishes to use in production
            Default value is current torch.backends.quantized.engine
    """

    # Keys for return dictionary
    BACKEND_KEY = "backend"
    PER_CHAN_SUPPORTED_KEY = "per_channel_quantization_supported"
    PER_CHAN_USED_KEY = "per_channel_quantization_used"

    # Default map for representing supported per channel quantization modules for different backends
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: dict[str, set[Any]] = {
        "fbgemm": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "qnnpack": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "onednn": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
        "x86": {
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nnqat.Linear,
            nnqat.Conv1d,
            nnqat.Conv2d,
            nnqat.Conv3d,
        },
    }

    def __init__(self, backend: str = torch.backends.quantized.engine):
        super().__init__()

        # store the backend information
        self.backend_chosen = backend
        self.supported_modules = set()
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[
                self.backend_chosen
            ]
        else:
            raise ValueError(
                f"Not configured to work with {self.backend_chosen}. Try a different default backend"
            )

    def get_detector_name(self) -> str:
        r"""returns the string name of this detector"""
        return "per_channel_detector"

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        per_channel_info = self._detect_per_channel_helper(model)

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in per_channel_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            per_chan_supported: bool = per_channel_info[module_fqn][
                self.PER_CHAN_SUPPORTED_KEY
            ]
            detector_qconfig_info.is_weight_per_channel = per_chan_supported
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, model: nn.Module) -> dict:
        r"""
        There is no observers inserted for the PerChannelDetector.

        Returns an empty dictionary since no observers are added or needed
        """
        return {}

    def _detect_per_channel_helper(self, model: nn.Module):
        r"""
        determines if per_channel quantization is supported in modules and submodules.

        Returns a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            model: The current module that is being checked to see if it is per_channel quantizable

        Returns dictionary mapping fqns to if per_channel quantization is possible
        """
        # create dict we will return
        per_channel_info: dict = {}

        # get the fully qualified name and check if in list of modules to include and list of modules to ignore
        for fqn, module in model.named_modules():
            is_in_include_list = any(
                isinstance(module, x) for x in self.supported_modules
            )

            # check if the module per_channel is supported
            # based on backend
            per_channel_supported = False

            if is_in_include_list:
                per_channel_supported = True

                # assert statement for MyPy
                q_config_file = module.qconfig
                if not isinstance(q_config_file, QConfig):
                    raise AssertionError("module.qconfig must be a QConfig")

                # this object should either be fake quant or observer
                q_or_s_obj = module.qconfig.weight.p.func()
                if not isinstance(q_or_s_obj, (FakeQuantize, ObserverBase)):
                    raise AssertionError(
                        "module.qconfig.weight must be a FakeQuantize or ObserverBase"
                    )

                per_channel_used = False  # will be true if found in qconfig

                if hasattr(
                    q_or_s_obj, "ch_axis"
                ):  # then we know that per_channel quantization used
                    # all fake quants have channel axis so need to check is_per_channel
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if (
                            hasattr(q_or_s_obj, "is_per_channel")
                            and q_or_s_obj.is_per_channel
                        ):
                            per_channel_used = True
                    elif isinstance(q_or_s_obj, ObserverBase):
                        # should be an observer otherwise
                        per_channel_used = True
                    else:
                        raise ValueError("Should be either observer or fake quant")

                per_channel_info[fqn] = {
                    self.PER_CHAN_SUPPORTED_KEY: per_channel_supported,
                    self.PER_CHAN_USED_KEY: per_channel_used,
                    self.BACKEND_KEY: self.backend_chosen,
                }

        return per_channel_info

    def generate_detector_report(self, model: nn.Module) -> tuple[str, dict[str, Any]]:
        r"""Checks if any Linear or Conv layers in the model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        Looks at q_config format and backend to determine if per_channel can be utilized.
        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support

        Args:
            model: The prepared and calibrated model we want to check if using per_channel

        Returns a tuple with two elements:
            String report of potential actions to improve model (if per_channel quantization is available in backend)
            Dictionary mapping per_channel quantizable elements to:
                whether per_channel quantization is supported by the backend
                if it is being utilized in the current model
        """

        # run the helper function to populate the dictionary
        per_channel_info = self._detect_per_channel_helper(model)

        # String to let the user know of further optimizations
        further_optims_str = (
            f"Further Optimizations for backend {self.backend_chosen}: \n"
        )

        optimizations_possible = False
        for fqn in per_channel_info:
            fqn_dict = per_channel_info[fqn]
            if (
                fqn_dict[self.PER_CHAN_SUPPORTED_KEY]
                and not fqn_dict[self.PER_CHAN_USED_KEY]
            ):
                optimizations_possible = True
                further_optims_str += (
                    f"Module {fqn} can be configured to use per_channel quantization.\n"
                )

        if optimizations_possible:
            further_optims_str += "To use per_channel quantization, make sure the qconfig has a per_channel weight observer."
        else:
            further_optims_str += "No further per_channel optimizations possible."

        # return the string and the dictionary form of same information
        return (further_optims_str, per_channel_info)


class DynamicStaticDetector(DetectorBase):
    r"""
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    Args:
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME = "model_report_pre_observer"
    DEFAULT_POST_OBSERVER_NAME = "model_report_post_observer"

    # naming conventions for stationary vs non-stationary data
    STATIONARY_STR = "stationary"
    NON_STATIONARY_STR = "non-stationary"

    # naming for activation
    INPUT_ACTIVATION_PREFIX = "input_activation_"
    OUTPUT_ACTIVATION_PREFIX = "output_activation_"

    # naming conventions for the keys of the return module info
    TOLERANCE_KEY = "dynamic_static_tolerance"
    DEFAULT_DYNAMIC_REC_KEY = "dynamic_recommended"
    PRE_OBS_COMP_STAT_KEY = INPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    POST_OBS_COMP_STAT_KEY = OUTPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    PRE_OBS_DATA_DIST_KEY = (
        INPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    )
    POST_OBS_DATA_DIST_KEY = (
        OUTPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    )
    IS_CURRENTLY_SUPPORTED_KEY = "is_dynamic_supported"

    # modules that are supported both dynamic and static for this report function
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = {nn.Linear}

    # modules that will be supported soon for both
    DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED = {nn.Conv1d, nn.Conv2d, nn.Conv3d}

    def __init__(self, tolerance=0.5):
        super().__init__()

        # set tolerance level and initialize a set to keep track of useful fqn locations
        self.tolerance = tolerance
        self.useful_observer_fqns: set[str] = set()

    def determine_observer_insert_points(
        self, prepared_fx_model: GraphModule
    ) -> dict[str, dict[str, Any]]:
        r"""
        Determines where observers need to be inserted for the Dynamic vs Static detector.
        For this detector, we want to place observers on either side of linear layers in the model.

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: dict[str, dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # make sure module is supported
            if self._is_supported(module, insert=True):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

                # add entry for post-observer
                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME

                obs_fqn_to_info[post_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),
                    DETECTOR_IS_POST_OBS_KEY: True,
                    DETECTOR_OBS_ARGS_KEY: (targeted_node,),
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""returns the string name of this detector"""
        return "dynamic_vs_static_detector"

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        dynamic_static_info = self._generate_dict_info(model)

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in dynamic_static_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            dynamic_static_recommended: bool = dynamic_static_info[module_fqn][
                self.DEFAULT_DYNAMIC_REC_KEY
            ]
            detector_qconfig_info.is_activation_dynamic = dynamic_static_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # check to see if module is of a supported type
        is_supported_type = any(
            isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED
        )

        # check if it will be supported
        future_supported_type = any(
            isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED
        )

        # supported
        supported = is_supported_type or future_supported_type

        # this is check for observer insertion
        if insert:
            return supported
        else:
            # this is for report gen and we also need to check if it contains observers
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME) and hasattr(
                module, self.DEFAULT_POST_OBSERVER_NAME
            )
            return supported and has_obs

    def _generate_dict_info(self, model: GraphModule) -> dict[str, Any]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """
        # store modules dynamic vs static information
        module_dynamic_static_info = {}

        # This for loop goes through the modules, and extracts all relevant information into module_dynamic_static_info
        #   This information primary includes whether the data distributions around a supported module is stationary or not
        #   Based on this, it is recorded whether dynamic or static quantization is recommended

        # loop through all submodules included nested ones
        for fqn, module in model.named_modules():
            # if module is Linear has the ModelReportObserver attached to it
            if self._is_supported(module):
                # get pre and post observers for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                post_obs = getattr(module, self.DEFAULT_POST_OBSERVER_NAME)

                # get the statistics for each module
                pre_stat = pre_obs.get_batch_to_epoch_ratio()
                post_stat = post_obs.get_batch_to_epoch_ratio()

                # record module, pre and post stat, and whether to do dynamic or static based off it
                # true if post observer data distribution is non-stationary, false if it's stationary
                dynamic_recommended = post_stat <= self.tolerance

                # specify the classifications for whether data distributions considered stationary or non-stationary
                pre_obs_dist_classif = (
                    self.STATIONARY_STR
                    if pre_stat > self.tolerance
                    else self.NON_STATIONARY_STR
                )
                post_obs_dist_classif = (
                    self.STATIONARY_STR
                    if post_stat > self.tolerance
                    else self.NON_STATIONARY_STR
                )

                # check if current support or future support
                is_supported_type = any(
                    isinstance(module, x)
                    for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED
                )

                # store the set of important information for this module
                module_info = {
                    self.TOLERANCE_KEY: self.tolerance,
                    self.DEFAULT_DYNAMIC_REC_KEY: dynamic_recommended,
                    self.PRE_OBS_COMP_STAT_KEY: pre_stat,
                    self.PRE_OBS_DATA_DIST_KEY: pre_obs_dist_classif,
                    self.POST_OBS_COMP_STAT_KEY: post_stat,
                    self.POST_OBS_DATA_DIST_KEY: post_obs_dist_classif,
                    self.IS_CURRENTLY_SUPPORTED_KEY: is_supported_type,
                }

                module_dynamic_static_info[fqn] = module_info

        return module_dynamic_static_info

    def generate_detector_report(
        self, model: GraphModule
    ) -> tuple[str, dict[str, Any]]:
        r"""
        Determines whether dynamic or static quantization is more appropriate for a given module.

        Takes advantage of the ModelReportObserver that records range information.
        Stationary distribution of data are strictly above tolerance level for the comparison statistic:

            S = average_batch_activation_range/epoch_activation_range

        Nonstationary distributions are below or at the tolerance level for this metric.

        If the distribution of data right after the module is non-stationary, recommend dynamic quantization
            Otherwise recommend static quantization

        This will then generate suggestions for dynamic vs static quantization focused around Linear.

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a tuple with two elements:
            String report of of whether dynamic or static quantization is recommended for certain modules
            Dictionary mapping modules with ModelReportObservers around them to:
                whether dynamic quantization is recommended
                their S metric of input to module
                whether input to module is stationary or non-stationary
                their S metric of output of module
                whether output of module is stationary or non-stationary
                the tolerance level to decided whether input/output is stationary or non-stationary
                whether it is currently supported or planned for the future
        """

        # get the dictionary of the information to format the string report
        module_dynamic_static_info = self._generate_dict_info(model)

        dynamic_vs_static_string = "Dynamic vs. Static Quantization suggestions: \n"

        modules_added: bool = False  # check to make sure at least 1 module added.

        dynamic_benefit = (
            " You will get more accurate results if you use dynamic quantization"
        )
        static_benefit = (
            " You can increase model efficiency if you use static quantization"
        )
        future_support_str = (
            ". This layer is not yet supported for dynamic quantization"
        )
        # This for loop goes through the information collected in module_dynamic_static_info and:
        #   Populates the string based report with the information from module_dynamic_static_info
        #   Compiles the complete report by appending relevant formatted strings

        for module_fqn in module_dynamic_static_info:
            # there is at least 1 module for suggestion
            modules_added = True
            module_info = module_dynamic_static_info[module_fqn]
            suggestion_string_template = (
                "For module {} it is suggested to use {} quantization because {}.\n"
            )

            # decide what string formatting values will be
            quantization_type = ""
            quantization_reasoning = "the distribution of data before {} is {} and the distribution after is {}."

            benefit_str = ""

            # strings for if dynamic quantized per tensor is needed
            recommend_per_tensor = (
                ". We recommend to add a {} before this module if it is static."
            )
            rec_lay_to_add = "dynamic quantize per tensor layer"
            dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
            dynamic_per_tensor_reasoning_string = " This is because the input to this module has a non-stationary distribution"

            # start composing explanation
            if module_info[self.DEFAULT_DYNAMIC_REC_KEY]:
                quantization_type = "dynamic"
                # check if currently supported or future supported
                benefit_str = dynamic_benefit
                if not module_info[self.IS_CURRENTLY_SUPPORTED_KEY]:
                    benefit_str += future_support_str
            else:
                quantization_type = "static"
                benefit_str = static_benefit

            # now set the quantization explanation string
            quantization_reasoning = (
                quantization_reasoning.format(
                    module_fqn,
                    module_info[self.PRE_OBS_DATA_DIST_KEY],
                    module_info[self.POST_OBS_DATA_DIST_KEY],
                )
                + benefit_str
            )

            # if we have a non-stationary input -> linear -> stationary we suggested static
            # however, we want to also recommend they add a dynamic quantize per tensor right if this change is made
            if (
                module_info[self.PRE_OBS_DATA_DIST_KEY] == self.NON_STATIONARY_STR
                and module_info[self.POST_OBS_DATA_DIST_KEY] == self.STATIONARY_STR
            ):
                quantization_reasoning = (
                    quantization_reasoning
                    + dynamic_per_tensor_string
                    + dynamic_per_tensor_reasoning_string
                )

            # format the overall suggestion string with the specific inputs
            module_suggestion_string = suggestion_string_template.format(
                module_fqn, quantization_type, quantization_reasoning
            )

            # append to overall suggestion
            dynamic_vs_static_string += module_suggestion_string

        if not modules_added:
            dynamic_vs_static_string += "No applicable layers for suggestions. Only linear and conv are valid.\n"

        # return the string as well as the dictionary of information
        return (dynamic_vs_static_string, module_dynamic_static_info)


class InputWeightEqualizationDetector(DetectorBase):
    r"""
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is suggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is suggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """

    SUPPORTED_MODULES: set[Callable] = {
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nnqat.Linear,
        nnqat.Conv1d,
        nnqat.Conv2d,
        nnqat.Conv3d,
    }

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"

    # weight / activation prefix for each of the below info
    WEIGHT_PREFIX = "weight_"
    ACTIVATION_PREFIX = "input_activation_"

    # string names for keys of info dictionaries
    PER_CHANNEL_MAX_KEY = "per_channel_max"
    PER_CHANNEL_MIN_KEY = "per_channel_min"
    GLOBAL_MAX_KEY = "global_max"
    GLOBAL_MIN_KEY = "global_min"

    # keys for return dict of recommendations
    RECOMMENDED_KEY = "input_weight_equalization_recommended"
    COMP_METRIC_KEY = "input_weight_channel_comparison_metrics"
    THRESHOLD_KEY = "input_weight_threshold"
    CHANNEL_KEY = "input_weight_channel_axis"

    # default weight and info strings
    WEIGHT_STR = "weight"
    INPUT_STR = "input"

    # default for what ratio we recommend input weight
    DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO = 0.4

    def __init__(self, ratio_threshold: float, ch_axis: int = 1):
        # ensure passed in inputs are valid
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError("Make sure threshold is > 0 and < 1")

        # initialize attributes based on args
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis

    def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # check to see if module is of a supported type
        is_supported_type = any(type(module) is x for x in self.SUPPORTED_MODULES)

        # this is check for observer insertion
        if insert:
            return is_supported_type
        else:
            # this is for report gen and we also need to check if it contains observers
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            return is_supported_type and has_obs

    def get_qconfig_info(self, model) -> dict[str, DetectorQConfigInfo]:
        r"""Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        # find the range of inputs
        input_values: dict[str, dict] = self._extract_input_info(model)

        # find the range of weights
        weight_values: dict[str, dict] = self._extract_weight_info(model)

        # calculate per_channel comparison statistic s_c
        comp_stats: dict[str, torch.Tensor] = self._generate_comparison_values(
            input_values, weight_values
        )

        # generate the return dictionary
        input_weight_equalization_info: dict[str, dict] = self._generate_dict_info(
            input_values, weight_values, comp_stats
        )

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in input_weight_equalization_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            input_weight_recommended: bool = input_weight_equalization_info[module_fqn][
                self.RECOMMENDED_KEY
            ]
            detector_qconfig_info.is_equalization_recommended = input_weight_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(
        self, prepared_fx_model: GraphModule
    ) -> dict[str, dict[str, Any]]:
        r"""Determines where observers need to be inserted for the Input Weight Equalization Detector.
        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            linear layers
            conv layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: dict[str, dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # check to see if module is of a supported type
            if self._is_supported(module, insert=True):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""Returns the name of this detector"""
        return "input_weight_equalization_detector"

    def _extract_input_info(self, model: GraphModule) -> dict[str, dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the input information for each observer returns it

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns (str) to a dict with keys:
            "input_activation_per_channel_max" : maps to the per_channel max values
            "input_activation_per_channel_min" : maps to the per_channel min values
            "input_activation_global_max" : maps to the global max recorded
            "input_activation_global_min" : maps to the global min recorded
        """

        # return dictionary mapping observer fqns to desired info
        input_info: dict[str, dict] = {}

        for fqn, module in model.named_modules():
            # if module is supported and it has a pre-observer
            if self._is_supported(module):
                # get pre observer for the module
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

                input_info[fqn] = {
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MAX_KEY: pre_obs.max_val,
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MIN_KEY: pre_obs.min_val,
                    self.ACTIVATION_PREFIX + self.GLOBAL_MAX_KEY: max(pre_obs.max_val),
                    self.ACTIVATION_PREFIX + self.GLOBAL_MIN_KEY: min(pre_obs.min_val),
                }

        return input_info

    def _extract_weight_info(self, model: GraphModule) -> dict[str, dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
        # return dictionary mapping observer fqns to desired info
        weight_info: dict[str, dict] = {}

        for fqn, module in model.named_modules():
            # if module is supported and it has a pre-observer
            if self._is_supported(module):
                # we don't need actual observer, just the module weights
                # calculate min and max vals
                device = module.weight.device
                min_val: torch.Tensor = torch.tensor([float("inf")], device=device)
                max_val: torch.Tensor = torch.tensor([float("-inf")], device=device)
                x_copy = module.weight
                x_dim = x_copy.size()

                new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
                new_axis_list[self.ch_axis] = 0
                new_axis_list[0] = self.ch_axis
                y = x_copy.permute(new_axis_list)

                # Need to match dtype of min/max because the updates to buffers
                # are done in place and types need to match for comparisons
                y = y.to(min_val.dtype)
                y = torch.flatten(y, start_dim=1)
                if min_val.numel() == 0 or max_val.numel() == 0:
                    min_val, max_val = torch.aminmax(y, dim=1)
                else:
                    min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                    min_val = torch.min(min_val_cur, min_val)
                    max_val = torch.max(max_val_cur, max_val)

                weight_info[fqn] = {
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val,
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val,
                    self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val),
                    self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val),
                }

        return weight_info

    def _calculate_range_ratio(
        self, info_dict: dict, info_str: str, module_fqn: str
    ) -> torch.Tensor:
        r"""
        Takes in an info dict and calculates the s_c matrix.

        Args:
            info_dict (dict): A dictionary of either input or weight range info
            info_str (str): A str describing whether currently looking at weight or input info
                Either "weight" or "input"
            module_fqn (str): The fqn of the module we are looking at

        Returns a tensor of values, where each value is the s_c stat for a different channel
        """
        # calculate the ratios of the info
        # get the prefix str
        prefix_str = (
            self.ACTIVATION_PREFIX if info_str == self.INPUT_STR else self.WEIGHT_PREFIX
        )

        per_channel_range = (
            info_dict[prefix_str + self.PER_CHANNEL_MAX_KEY]
            - info_dict[prefix_str + self.PER_CHANNEL_MIN_KEY]
        )
        global_range = (
            info_dict[prefix_str + self.GLOBAL_MAX_KEY]
            - info_dict[prefix_str + self.GLOBAL_MIN_KEY]
        )

        if global_range == 0:
            range_zero_explanation = "We recommend removing this channel as it doesn't provide any useful information."
            raise ValueError(
                f"The range of the {info_str} data for module {module_fqn} is 0, "
                f"which means you have a constant value channel. {range_zero_explanation}"
            )

        ratio = per_channel_range / global_range

        return ratio

    def _generate_comparison_values(
        self, input_info: dict, weight_info: dict
    ) -> dict[str, torch.Tensor]:
        r"""
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
        # create return dictionary for each observer
        module_fqn_to_channel: dict[str, torch.Tensor] = {}

        # for each module (both passed in dicts should have same keys)
        for module_fqn in input_info:
            # raise error if not in weight info
            if module_fqn not in weight_info:
                raise KeyError(
                    f"Unable to find weight range stats for module {module_fqn}"
                )

            # calculate the ratios of the weight info and input info
            weight_ratio = self._calculate_range_ratio(
                weight_info[module_fqn], self.WEIGHT_STR, module_fqn
            )
            input_ratio = self._calculate_range_ratio(
                input_info[module_fqn], self.INPUT_STR, module_fqn
            )

            # if mismatched size, because of grouping, we want to replicate weight enough times
            weight_channels = len(weight_ratio)
            input_channels = len(input_ratio)
            if weight_channels != input_channels:
                # we try to replicate
                if input_channels % weight_channels != 0:
                    raise AssertionError(
                        "input channels should be divisible by weight channels."
                    )
                # get replication factor
                rep_factor: int = input_channels // weight_channels

                # weight ratio is (n,), input ratio is (k,), we just 
```



## High-Level Overview

r"""    This class contains the QConfig information for a single module.    The list of variables / values this contains can grow depending on the    extensibility of the qconfig mapping feature set but this currently includes:    - if activation observer is dynamic    - if weight observer is per channel    Args:

This Python file contains 11 class(es) and 42 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DetectorQConfigInfo`, `DetectorBase`, `PerChannelDetector`, `DynamicStaticDetector`, `InputWeightEqualizationDetector`, `OutlierDetector`

**Functions defined**: `__init__`, `generate_quantization_qconfig`, `generate_equalization_qconfig`, `__init__`, `determine_observer_insert_points`, `get_detector_name`, `get_qconfig_info`, `_get_targeting_node`, `generate_detector_report`, `__init__`, `get_detector_name`, `get_qconfig_info`, `determine_observer_insert_points`, `_detect_per_channel_helper`, `generate_detector_report`, `__init__`, `determine_observer_insert_points`, `get_detector_name`, `get_qconfig_info`, `_is_supported`

**Key imports**: ABC, abstractmethod, Callable, Any, torch, torch.ao.nn.qat as nnqat, torch.nn as nn, FakeQuantize, GraphModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/fx/_model_report`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`: ABC, abstractmethod
- `collections.abc`: Callable
- `typing`: Any
- `torch`
- `torch.ao.nn.qat as nnqat`
- `torch.nn as nn`
- `torch.ao.quantization.fake_quantize`: FakeQuantize
- `torch.ao.quantization.fx.graph_module`: GraphModule


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/ao/quantization/fx/_model_report`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`model_report.py_docs.md`](./model_report.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`model_report_visualizer.py_docs.md`](./model_report_visualizer.py_docs.md)
- [`model_report_observer.py_docs.md`](./model_report_observer.py_docs.md)


## Cross-References

- **File Documentation**: `detector.py_docs.md`
- **Keyword Index**: `detector.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
