from ruamel.yaml import YAML

from collections.abc import Mapping
from collections import OrderedDict
from textwrap import dedent
from io import StringIO
import logging
from pathlib import Path
import joblib as jl
import jax.numpy as jnp
import numpy as np


def _create_yaml():
    """Create a Ruamel round trip parser and dumper."""
    _yaml = YAML(typ="rt")
    _yaml.indent(mapping=4, sequence=2, offset=2)
    _yaml.default_flow_style = False

    def _represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "null")

    _yaml.representer.add_representer(type(None), _represent_none)

    def _represent_ordered(self, data):
        return self.represent_mapping("tag:yaml.org,2002:map", data)

    _yaml.representer.add_representer(OrderedDict, _represent_ordered)

    return _yaml


_yaml = _create_yaml()


def loads(yml_string):
    """Create a Ruamel round trip map values from a yaml string."""
    yml_string = dedent(yml_string)
    if yml_string == "":
        return {}
    return _yaml.load(yml_string.strip())


def dumps(yml_dict):
    """Create a yaml string from a Ruamel round trip map."""
    with StringIO() as stream:
        _yaml.dump(yml_dict, stream)
        return stream.getvalue()


def save_project_config(path, config, write_calib=False):
    """Save project config, separating yaml values and calibration data.

    Parameters
    ----------
    path : pathlib.Path
    config : dict"""
    if isinstance(path, str):
        path = Path(path)

    if write_calib:
        calib = load_calibration_data(config["calibration_file"])
        calib.update(_extract_calibration_data(config))
        save_calibration_data(config["calibration_file"], calib)

    without_calib = _map_ordered(config, _drop_calibration_data)
    save_config(path, without_calib)


def save_config(path, config):
    """Save a config to a file.

    Parameters
    ----------
    path : pathlib.Path
    config : dict"""
    with open(str(path), "w") as f:
        _yaml.dump(config, f)


def load_project_config(path):
    """Load a config from a file.

    Parameters
    ----------
    path : pathlib.Path
    """
    if isinstance(path, str):
        path = Path(path)
    with open(str(path), "r") as f:
        cfg = _yaml.load(f)
    # load calibration data and insert into config sections
    cfg = _add_calibration_data(
        cfg, load_calibration_data(cfg["calibration_file"])
    )
    return cfg


def _add_calibration_data(config, calibration_data):
    """Add calibration data to a config."""
    for k, v in config.items():
        if isinstance(v, Mapping):
            if k in calibration_data:
                v["calibration_data"] = calibration_data[k]
            else:
                v["calibration_data"] = {}
    return config


def load_config(path):
    """Load a config from a file.

    Parameters
    ----------
    path : pathlib.Path
    """
    if isinstance(path, str):
        path = Path(path)
    with open(str(path), "r") as f:
        return _yaml.load(f)


def _drop_calibration_data(config_val):
    if isinstance(config_val, Mapping):
        return _drop_ordered(config_val, "calibration_data")
    else:
        return config_val


def _extract_calibration_data(config):
    """Extract calibration data from a config."""
    return {
        k: v["calibration_data"]
        for k, v in config.items()
        if isinstance(v, Mapping) and "calibration_data" in v
    }


def _drop_ordered(d, key):
    """Drop a key from an ordered dictionary."""
    return OrderedDict(tuple((k, v) for k, v in d.items() if k != key))


def _map_ordered(d, func):
    """Map a function over the values of an ordered dictionary."""
    return OrderedDict((k, func(v)) for k, v in d.items())


def load_model_config(path):
    """Load a model config from a file.

    Parameters
    ----------
    path : pathlib.Path
    """
    if isinstance(path, str):
        path = Path(path)
    # load yaml values, including main project config
    with open(str(path), "r") as f:
        model_cfg = _yaml.load(f)
    with open(model_cfg["project"], "r") as f:
        model_cfg.update(_yaml.load(f))
    # load calibration data and insert into config sections
    model_cfg = _add_calibration_data(
        model_cfg, load_calibration_data(model_cfg["calibration_file"])
    )
    return model_cfg


def save_model_config(path, config, write_calib=False):
    """Save a model config to a file.

    Parameters
    ----------
    path : pathlib.Path
    config : dict"""
    if isinstance(path, str):
        path = Path(path)
    # select only data relevant to the model configuration
    model_cfg = OrderedDict(
        (
            ("project", config["project"]),
            ("pose", config["pose"]),
            ("morph", config["morph"]),
            ("fit", config["fit"]),
        )
    )
    # save YAML config without calibration data
    without_calib = _map_ordered(model_cfg, _drop_calibration_data)
    save_config(path, without_calib)
    # save project-wide calibration data if requested
    if write_calib:
        calib = load_calibration_data(config["calibration_file"])
        calib.update(_extract_calibration_data(model_cfg))
        save_calibration_data(config["calibration_file"], calib)


def load_calibration_data(path):
    """Load calibration data from a file.

    Parameters
    ----------
    path : pathlib.Path
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return {}
    return jl.load(str(path))


def save_calibration_data(path, data):
    """Save calibration data to a file.

    Parameters
    ----------
    path : pathlib.Path
    data : dict
    """
    if isinstance(path, str):
        path = Path(path)
    jl.dump(data, str(path))


def get_entries(config, keys):
    """Get the entries of a config.

    Parameters
    ----------
    config : dict
    keys : list of str
    """
    return OrderedDict((k, config[k]) for k in keys)


def get_values(config, keys):
    """Get ordered subset of values of a config.

    Parameters
    ----------
    config : dict
    keys : list of str
    """
    return [config[k] for k in keys]


def deepen(flat_dict):
    """Convert a patially flat dictionary to a nested one.

    Given keys {'a.b.c': 1}, will return {'a': {'b': 1}}.
    """
    flat_dict = flatten(flat_dict)
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = nested_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested_dict


def flatten(nested_dict):
    """Convert a nested dictionary to a flat one.

    Given keys {'a': {'b': 1}, 'a.c': 2}, will return {'a.b': 1, 'a.c': 2}.

    In the case of conflicting paths, behavior is not defined.
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            subdict = flatten(value)
            for subkey, subvalue in subdict.items():
                flat_dict[key + "." + subkey] = subvalue
        else:
            flat_dict[key] = value
    return flat_dict


def _recursive_update(defaults, new_vals, add, root_path):
    ret = {k: v for k, v in defaults.items() if k not in new_vals}
    for k, v in new_vals.items():
        k_path = root_path + (k,)
        if k not in defaults:
            flat_path = ".".join(k_path)
            if add == "warn":
                logging.warning(f"Adding new key {flat_path} to config")
            elif add == "raise" or add is False:
                raise KeyError(
                    f"Key {flat_path} not present in defaults, cannot add."
                )
        if isinstance(v, dict) or isinstance(v, Mapping):
            ret[k] = _recursive_update(
                defaults.get(k, {}), v, add=add, root_path=k_path
            )
        else:
            ret[k] = v
    return ret


def recursive_update(defaults, new_vals, add="warn"):
    """Copy a nested dictionary with new values.

    Parameters
    ----------
    defaults : dict
    new_vals : dict
    add : str or bool, default "warn"
        Behavior when encountering new keys not in defaults. If "warn", will add
        new keys to defaults and log a warning. If "raise", or False, will
        raise an error if new keys are present. If True will add new keys
        silently.
    """
    return _recursive_update(defaults, new_vals, add, root_path=())


def recursive_eq(a, b):
    """Check if two nested dictionaries are equal."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(recursive_eq(a[k], b[k]) for k in a.keys())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(recursive_eq(x, y) for x, y in zip(a, b))
    else:
        if (
            isinstance(a, jnp.ndarray)
            or isinstance(a, np.ndarray)
            or hasattr(a, "shape")
        ) and (
            isinstance(b, jnp.ndarray)
            or isinstance(b, np.ndarray)
            or hasattr(b, "shape")
        ):
            return (a.shape == b.shape) and jnp.allclose(a, b)
        try:
            return a == b
        except:
            logging.error(f"cannot compare {type(a)} and {type(b)}")
            raise
