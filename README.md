# GPX

![GPX example on California housing dataset](https://raw.githubusercontent.com/yuyay/gpx/image/california_example.png)

GPX is a Gaussian process regression model that can output the feature contributions to the prediction for each sample, which is implemented based on the following paper:  
**Yuya Yoshikawa, and Tomoharu Iwata. "[Gaussian Process Regression With Interpretable Sample-Wise Feature Weights.](https://ieeexplore.ieee.org/abstract/document/9646444)" IEEE Transactions on Neural Networks and Learning Systems (2021).**

GPX has the following characteristics:
- High accuracy: GPX can achieve comparable predictive accuracy to standard Gaussian process regression models.
- Explainability: GPX can output feature contributions with uncertainty for each sample. We showed that the feature contributions are more appropriate qualitatively and quantitatively than the existing explanation methods, such as LIME and SHAP, etc. 

## Installation 
The pytorch-gpx package is on PyPI. Simply run:
```bash
pip install pytorch-gpx
```
Or clone the repository and run:
```bash
pip install .
```

## Usage
The pytorch-gpx package provides scikit-learn-like API for training, prediction, and evaluation of GPX models.

```python
from sklearn.metrics import mean_squared_error
from gpx import GPXRegressor

'''Training
X_tr: input data (numpy array), with shape of (n_samples, n_X_features)
y_tr: target variables (numpy array), with shape of (n_samples,)
Z_tr: simplified input data (numpy array), with shape of (n_samples, n_Z_features). The same as X_tr is OK.
'''
model = GPXRegressor().fit(X_tr, y_tr, Z_tr)

'''Prediction
y_mean: the posterior mean of target variables
y_conv: the posterior variance of target variables
w_mean: the posterior mean of weights
w_conv: the posterior variance of weights
'''
y_mean, y_cov, w_mean, w_cov = model.predict(X_te, Z_te, return_weights=True)

'''Evaluation'''
mse = mean_squared_error(y_te, y_mean)
print("Test MSE = {}".format(mse))
```

For more usage examples, please see the below.
- [Regression on California housing price dataset (tabular data)](notebooks/california_regression.ipynb)
- [Label regression on binary-class hand-written digits dataset (image data)](notebooks/digits_visualization.ipynb)

## Citation
If you use this repo, please cite the following paper.

```bibtex
@article{yoshikawa2021gpx,
  title={Gaussian Process Regression With Interpretable Sample-Wise Feature Weights},
  author={Yoshikawa, Yuya and Iwata, Tomoharu},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```

## License
Please see [LICENSE.txt](./LICENSE.txt).

## Acknowledgment
This work was supported by the Japan Society for the Promotion of Science (JSPS) KAKENHI under Grant 18K18112.
