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

# Defining a better least-squares function 
def ls(A, B, rcond=None):
    return jnp.linalg.lstsq(A,B, rcond=rcond)[0]

# Pydantic imports
import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, computed_field, PrivateAttr, SkipValidation
from abc import ABC, abstractmethod
from typing import Callable 
