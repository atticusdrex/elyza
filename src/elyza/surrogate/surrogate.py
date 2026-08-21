from elyza.util.imports import * 
from elyza.core.data import * 
from elyza.util.helpers import ensure_2d

'''
~---------------------------------~
|  Base Surrogate Modeling Class  |
~---------------------------------~
This is the base surrogate modeling class from which all the multifidelity functionality is built. It basically just defines a common callable structure so that the multifidelity surrogate models don't have to be tailor-made to specific types of ML models. It loosely follows the scikit learn regressor structure to act as a wrapper for general types of ML models. 

model = MyModel() 

Suppose we train my model using: 

model.train_model(
    X_data = X, 
    Y_data = Y,
    initial_params = None, 
    momentum = 0.9 
)

We would then define the Surrogate() class using: 

class MySurrogate(Surrogate): 
    def fit(
        self, 
        X, Y, 
        momentum = 0.9, 
        initial_params = None
    ): 
        self.model.train_model(
            X_data = X, 
            Y_data = Y, 
            initial_params = initial_params, 
            momentum = momentum 
        )

    def predict(
        self, 
        X, 
        full_cov = False
    ): 

        self.model.make_prediction(
            X_data = X, 
            full_cov = full_cov 
        )

When a child class does not implement all the methods (e.g., a deep learning class may not implement a sample() or update() method), the surrogate model base class defaults to raising a "NotImplementedError"
'''
class Surrogate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def fit(
        self, 
        X: np.ndarray | jax.Array, 
        Y: np.ndarray | jax.Array, 
        **kwargs
    ) -> None:
        """Fit the surrogate model to training data."""
        raise NotImplementedError("This feature is not implemented yet.")


    def predict(
        self, 
        X: np.ndarray | jax.Array, 
        **kwargs 
    ) -> tuple[jax.Array]:
        """Return predictive mean (and optionally variance) at new points."""
        raise NotImplementedError("This feature is not implemented yet.")


    def sample(self, X, n_samples: int = 1, random_state: int = 0) -> jax.Array:
        """Draw samples from the posterior/predictive distribution."""
        raise NotImplementedError("This feature is not implemented yet.")


    def update(self, X, Y) -> None:
        """Update the model with new observations (e.g. online/incremental fitting)."""
        raise NotImplementedError("This feature is not implemented yet.")


'''
wrapper class for working with supervised learning datasets 
'''
class SupervisedDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_data : list[jax.Array] = Field(description = "list of in-order inputs and the data associated with those inputs")
    output_data : jax.Array = Field(description = "an array of the corresponding model outputs associated with these inputs")
    noise_var : float = Field(default = 0.0, description = "variance of Gaussian white noise in the output data")

    # concatenate the inputs into one big array 
    def concatenate_inputs(self):
        return jnp.concatenate(self.input_data, axis=1) 

    def model_post_init(self, __context):
        self.output_data = ensure_2d(self.output_data)

    def update(self, *new_inputs : list[jax.Array], new_outputs : jax.Array):
        # adding the new inputs to the existing input 
        for i, (existing_input, new_input) in enumerate(zip(self.input_data, new_inputs)):
            self.input_data[i] = jnp.concatenate((existing_input, ensure_2d(new_input)), axis=0)

        # adding the new outputs to the existing outputs 
        self.output_data = jnp.concatenate((self.output_data, ensure_2d(new_outputs)), axis=0)




        
        



        





        


        
            



        
        


    
