data {
  int<lower=0> n_dogs;
  int<lower=0> n_trials;
  array[n_dogs, n_trials] int<lower=0, upper=1> y;
}
transformed data {
  int N = n_dogs * n_trials;
  matrix[N, 2] X;
  array[N] int yy;
  for (j in 1 : n_dogs) {
    real n_avoid = 0;
    real n_shock = 0;
    for (t in 1 : n_trials) {
      if (t > 1) {
        n_avoid += 1 - y[j, t - 1];
        n_shock += y[j, t - 1];
      }
      X[(j - 1) * n_trials + t, 1] = n_avoid;
      X[(j - 1) * n_trials + t, 2] = n_shock;
      yy[(j - 1) * n_trials + t] = y[j, t];
    }
  }
}
parameters {
  vector[3] beta;
}
model {
  beta ~ normal(0, 100);
  yy ~ bernoulli_logit_glm(X, beta[1], beta[2:3]);
}
