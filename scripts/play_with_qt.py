import numpy as np
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
from scipy import stats

# Generate some sample data
X = np.random.uniform(-5,5,1000)  # Example data

# Fit the QuantileTransformer
qt = QuantileTransformer(n_quantiles=10, output_distribution='normal')
qt.fit(X.reshape(-1, 1))

Xt = qt.transform(X.reshape(-1, 1))

# Get the quantiles
quantiles = qt.quantiles_

print(quantiles)


def _transform_col(X_col, quantiles, inverse):
        """Private function to transform a single feature."""

        quantiles = np.squeeze(quantiles)

        BOUNDS_THRESHOLD = 1e-7
        n_quantiles_ = np.shape(quantiles)[0]
        references_ = np.linspace(0, 1, n_quantiles_, endpoint=True)

        output_distribution = "normal"

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]            
            X_col = stats.norm.cdf(X_col)

        lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
        upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
  

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]

        if not inverse:
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        if not inverse:
            with np.errstate(invalid="ignore"):
                if output_distribution == "normal":
                    X_col = stats.norm.ppf(X_col)
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)

        return X_col

X_new = X.copy()
X_newt = _transform_col(X_new, quantiles, inverse=False)


plt.subplot(2,3,1)
plt.hist(X, bins=50)
plt.subplot(2,3,2)
plt.hist(Xt, bins=50)
plt.subplot(2,3,3)
plt.hist(X_newt, bins=50)
plt.subplot(2,3,4)

plt.hist([Xt[:,0], X_newt], bins=50, histtype='step')
plt.subplot(2,3,5)
plt.hist(_transform_col(X_newt, quantiles, inverse=True), bins=50, histtype='step')
plt.subplot(2,3,6)
plt.hist([X, _transform_col(X_newt, quantiles, inverse=True)], bins=50, histtype='step')

plt.show()
