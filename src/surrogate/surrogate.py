from util.imports import * 
from core.data import * 

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


    


        
        



        





        


        
            



        
        


    
