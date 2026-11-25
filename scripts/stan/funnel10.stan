parameters {
  real v;
  vector[9] x;
}
model {
  v ~ normal(0, 3);
  x ~ normal(0, exp(v / 2));
}
