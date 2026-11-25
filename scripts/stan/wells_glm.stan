data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> switched;
  vector[N] dist;
}
transformed data {
  matrix[N, 1] X = to_matrix(dist);
}
parameters {
  vector[2] beta;
}
model {
  switched ~ bernoulli_logit_glm(X, beta[1], beta[2:2]);
}
