
from sklearn.mixture import GaussianMixture


def create_gmm(
    n_components=38,
    covariance_type="diag",
    max_iter=200,
    n_init=3,
    reg_covar=1e-5,
    random_state=42,
):
    """Create the configured Gaussian Mixture Model."""

    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    )


def train_gmm(X_train, **model_params):
    """Train GMM exclusively on legitimate training transactions."""

    model = create_gmm(**model_params)
    model.fit(X_train)

    return model


def calculate_anomaly_scores(model, X):
    """
    Calculate negative log-likelihood anomaly scores.
    Higher scores indicate more anomalous transactions.
    """

    return -model.score_samples(X)


def get_model_info(model):
    """Return key information about the trained GMM."""

    return {
        "components": model.n_components,
        "covariance_type": model.covariance_type,
        "converged": model.converged_,
        "iterations": model.n_iter_,
        "lower_bound": model.lower_bound_,
    }
