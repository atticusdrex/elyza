---
file_format: mystnb
kernelspec:
  name: python3
---

# Linear Models

`elyza.surrogate.linear` provides two linear surrogates: {class}`~elyza.surrogate.linear.linreg.Ridge`,
closed-form L2-regularized regression, and {class}`~elyza.surrogate.linear.gmm.GMMRegression`,
a weight-space Bayesian linear model whose coefficients follow a Gaussian
mixture. Both take a design matrix of *features* rather than raw inputs, which
is where `elyza.util.preprocessing`'s feature-construction objects come in --
this page uses two of them, {class}`~elyza.util.preprocessing.OrthonormalScaler`
and {class}`~elyza.util.preprocessing.KernelFeatures`, to build those features.

Every example below runs as-is against a clean checkout.

## 1. Ridge Regression

{class}`~elyza.surrogate.linear.linreg.Ridge` fits `beta` via the regularized
normal equations `(XᵀX + l2_reg·I)⁻¹XᵀY` -- no optimizer needed. Here we fit a
2-d saddle-shaped function to show it off with a prediction heatmap:

```{code-cell} python
import jax.numpy as jnp
import jax.random as jrand

from elyza.core import Uniform, Evaluator

x = Uniform(
    name="x",
    dim=2,
    lower=jnp.array([-2.0, -2.0]),
    upper=jnp.array([2.0, 2.0]),
)
y = Evaluator(
    name="saddle",
    inputs=[x],
    output_dim=1,
    evaluation_func=lambda x: x[0]**2 - x[1]**2 + 0.5 * x[0] * x[1],
)

X_train = x.sample(jrand.PRNGKey(0), 300)
Y_train = y.evaluate(X_train) + 5e-2 * jrand.normal(jrand.PRNGKey(1), shape=(X_train.shape[0], 1))
```

`Ridge` penalizes every feature by the same `l2_reg`, so features on
mismatched scales get penalized unevenly. {class}`~elyza.util.preprocessing.OrthonormalScaler`
whitens the raw inputs via a truncated SVD -- centering them and rotating/
rescaling so the transformed columns are uncorrelated with unit variance --
before {class}`~elyza.util.preprocessing.PolynomialFeatures` expands them into
the quadratic basis `Ridge` actually fits against:

```{code-cell} python
from elyza.util.preprocessing import OrthonormalScaler, PolynomialFeatures
from elyza.surrogate.linear import Ridge

scaler = OrthonormalScaler()
X_white = scaler.fit_transform(X_train)

poly = PolynomialFeatures(degree=2)
F_train = poly.fit_transform(X_white)

ridge = Ridge(l2_reg=1e-2)
ridge.fit(F_train, Y_train)
```

Predicting over a grid and reshaping back to 2-d lets us compare the fitted
surface side-by-side against the true (noise-free) function it was trained
on:

```{code-cell} python
from matplotlib.pyplot import *
%matplotlib inline

n_grid = 60
x1 = jnp.linspace(-2.0, 2.0, n_grid)
x2 = jnp.linspace(-2.0, 2.0, n_grid)
xx1, xx2 = jnp.meshgrid(x1, x2)
X_test = jnp.stack([xx1.ravel(), xx2.ravel()], axis=1)

Y_true = y.evaluate(X_test).reshape(n_grid, n_grid)
F_test = poly.transform(scaler.transform(X_test))
Y_pred = ridge.predict(F_test).reshape(n_grid, n_grid)

zmin, zmax = float(min(Y_true.min(), Y_pred.min())), float(max(Y_true.max(), Y_pred.max()))

fig, axes = subplots(1, 2, figsize=(12, 5.5), dpi=110, subplot_kw={"projection": "3d"})

for ax, Z, subplot_title in zip(axes, [Y_true, Y_pred], ["True Surface", "Ridge Regression Prediction"]):
    surf = ax.plot_surface(xx1, xx2, Z, cmap="viridis", vmin=zmin, vmax=zmax, edgecolor="none", antialiased=True, alpha=0.95)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("y")
    ax.set_zlim(zmin, zmax)
    ax.set_title(subplot_title)
    ax.view_init(elev=25, azim=-60)

axes[1].scatter(X_train[:, 0], X_train[:, 1], Y_train.ravel(), s=6.0, color="red", alpha=0.4, depthshade=True)

fig.colorbar(surf, ax=axes, shrink=0.6, pad=0.05, label="y")
fig.suptitle("Ridge Regression: True vs. Predicted Surface")
show()
```

## 2. Gaussian Mixture Regression

{class}`~elyza.surrogate.linear.gmm.GMMRegression` puts a Gaussian-mixture
prior over the coefficients `beta` of a linear model `y = X @ beta`, so
predictions carry uncertainty in addition to a point estimate. Unlike
`Ridge`, it needs {mod}`~elyza.optim` to fit -- it maximizes the exact
Gaussian-mixture log-likelihood of the training targets, since a linear map
of a Gaussian mixture is itself a Gaussian mixture. Here `X` comes from
{class}`~elyza.util.preprocessing.KernelFeatures`, which maps a raw input to
its kernel evaluations against a fixed set of centers -- the "unique
preprocessing" analog of a radial-basis-function expansion:

```{code-cell} python
xg = Uniform(name="xg", dim=1, lower=0.0, upper=1.0)
yg = Evaluator(name="yg", inputs=[xg], output_dim=1, evaluation_func=lambda x: jnp.sin(2 * jnp.pi * x)**2)

X_train_g = xg.sample(jrand.PRNGKey(42), 1000)
Y_train_g = yg.evaluate(X_train_g) + 6e-2 * jrand.normal(jrand.PRNGKey(43), shape=X_train_g.shape)

# adding the negated y data to the dataset 
X_train_g = jnp.concatenate((X_train_g, X_train_g), axis=0) 
Y_train_g = jnp.concatenate((Y_train_g, -Y_train_g), axis=0)
```

```{code-cell} python
from elyza.util.preprocessing import KernelFeatures
from elyza.surrogate.gp import ARD
from elyza.surrogate.linear.gmm import GMMRegression
from elyza.optim import ADAM, ADAMOptions

features = KernelFeatures(input_dim=1, kernel_cls=ARD, eps=1e-12)
centers = jnp.linspace(0.0, 1.0, 50).reshape(-1, 1)
F_train_g = features.fit_transform(X_train_g, jnp.array([1.0, 1e-2]), centers)

model = GMMRegression(input_dim=50, n_dist=2, scale=1e-3, random_state=42)
model.set_optimizer(ADAM, ADAMOptions(lr=1e-3, epochs=10000, beta1=0.95, verbose=False, unroll=25))
model.fit(F_train_g, Y_train_g)
```

Just like the GP surrogates, `predict` returns a mean and variance, so the
predictive uncertainty can be plotted as a confidence band around the fit:

```{code-cell} python
X_test_g = jnp.linspace(0.0, 1.0, 1000).reshape(-1, 1)
F_test_g = features.transform(X_test_g)
ysamp = model.sample(
    jrand.PRNGKey(45), F_test_g, n_samples = 100
)

figure(figsize=(6, 4), dpi=100)
for i in range(ysamp.shape[1]): 
    if i == 0:
        plot_label = "GMM predictions"
    else:
        plot_label = None 
    scatter(X_test_g.ravel(), ysamp[:,i], s = 5.0, marker = ".", color = 'green', alpha = 0.01, label = plot_label)
    
plot(X_test_g.ravel(), yg.evaluate(X_test_g).ravel(), color="black", linestyle="dotted", label="Target function")
plot(X_test_g.ravel(), -yg.evaluate(X_test_g).ravel(), color="black", linestyle="dotted")
scatter(X_train_g.ravel(), Y_train_g.ravel(), s=5.0, color="red", alpha=0.1, label="training data")
xlabel("x")
ylabel("y")
title("GMMRegression Predictive Uncertainty")
legend()
show()
```
