data {
  int<lower=0> n_dogs;
  int<lower=0> n_trials;
  array[n_dogs, n_trials] int<lower=0, upper=1> y;
}
transformed data {
  matrix[n_dogs, n_trials] n_avoid;
  matrix[n_dogs, n_trials] n_shock;
  for (j in 1 : n_dogs) {
    n_avoid[j, 1] = 0;
    n_shock[j, 1] = 0;
    for (t in 2 : n_trials) {
      n_avoid[j, t] = n_avoid[j, t - 1] + 1 - y[j, t - 1];
      n_shock[j, t] = n_shock[j, t - 1] + y[j, t - 1];
    }
  }
}
parameters {
  vector[3] beta;
}
model {
  beta ~ normal(0, 100);
  for (i in 1 : n_dogs) {
    for (j in 1 : n_trials) {
      y[i, j] ~ bernoulli_logit(beta[1] + beta[2] * n_avoid[i, j] + beta[3] * n_shock[i, j]);
    }
  }
}
