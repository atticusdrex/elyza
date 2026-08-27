"""Centralized third-party imports shared across the ``elyza`` package.

Other modules do ``from elyza.util.imports import *`` to pull in numpy,
JAX, pydantic, and standard-library utilities with one line, so that
JAX's 64-bit configuration is applied exactly once, consistently.
"""

# Numpy Imports
import numpy as np

# Math imports
import math
from math import pi

# TQDM
from tqdm import tqdm

# Copy functionality
from copy import copy, deepcopy

# Jax imports
import jax
import jax.numpy as jnp
from jax import value_and_grad, vmap, jit
import jax.random as jrand
from jax import flatten_util
import jax.numpy.linalg as jla
from jax.scipy.linalg import cho_solve

# 64-bit operation
try:
    jax.config.update("jax_enable_x64", True)
except:
    print("Jax 64 bit is not available on your CPU!")

# Pydantic imports
import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, computed_field, PrivateAttr, SkipValidation
from abc import ABC, abstractmethod
from typing import Callable
