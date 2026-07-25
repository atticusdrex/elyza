# %%
from surrogate.gp.gp import GaussianProcess 
from surrogate.gp.kernel import ARD
from surrogate.gp.mean import Linear 

from util.imports import * 

from matplotlib.pyplot import * 


def sinusoidal_test(n_points = 100):
    y = lambda x: jnp.sin(2*pi*x[0]) + jnp.cos(2*pi*x[1])

    X = jrand.uniform(
        jrand.PRNGKey(42), 
        shape = (n_points,2)
    )

    Y = y(X.T)

    model = GaussianProcess(
        input_dim = 2, 
        kernel_cls = ARD, 
        mean_cls = Linear, 
        calibrate_noise = True, 
        noise_var = 1e-8, 
        eps = 1e-2, 
        max_cond = 1e5, 
        verbose = True
    )

    model.fit(
        X, Y, solver = 'adam', 
        learning_rate = 1e-2, 
        steps = 1000, 
        beta1 = 0.9, 
        beta2 = 0.999
    )

    X_new = jrand.uniform(
        jrand.PRNGKey(43), 
        shape = (50,2)
    )
    Y_new = y(X_new.T)
    
    model.update(
        X_new, Y_new
    )

    Ymean, Ycov = model.predict(X, full_cov = False)

    return jnp.max(jnp.abs(Ymean - Y)) <= 0.1






if __name__ == "__main__":
    print("Sinusoidal Test: ", sinusoidal_test())