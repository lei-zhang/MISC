// inspired by http://rpubs.com/dahtah/psychfun

data {
  int<lower=1> nS;                   // nSubject
  int<lower=1> nT;                   // nTrial
  vector<lower=0,upper=1>[nT] intensity;  // stimulus (difficulty / intensity)
  int<lower=0,upper=1> resp[nS,nT];  // response
}

transformed data{
    real max_lambda;    
    max_lambda = 0.1;
}

parameters {  
  // group-level parameters
  vector[3] mu_p;  
  vector<lower=0>[3] sigma;

  // raw individual-level parameters (for Matt Trick)
  vector[nS] eta_p;     // the indecision point (50%)
  vector[nS] theta_p;   // the inverse slope; lower sigma infers greater slope  
  vector[nS] lambda_p;  // a lapse rate: random choice every once in a while (Wichmann & Hill, 2001) 
}
transformed parameters {
  // individual-level parameters
  vector[nS] eta; 
  vector<lower=0>[nS] theta;
  vector<lower=0,upper=max_lambda>[nS] lambda;

  eta    = mu_p[1] + sigma[1] * eta_p;
  theta  = exp(mu_p[2] + sigma[2] * theta_p);
  lambda = Phi_approx(mu_p[3] + sigma[3] * lambda_p) * max_lambda;
}
model {  
  mu_p     ~ normal(0,1);
  sigma    ~ cauchy(0,3);

  eta_p    ~ normal(0,1);
  theta_p  ~ normal(0,1);
  lambda_p ~ normal(0,1);

  for (s in 1:nS) {
    vector[nT] delta;

    delta = (intensity - eta[s]) / theta[s];
    resp[s] ~ bernoulli( (1-lambda[s]) * Phi(delta) + lambda[s]/2 );
  }

}
generated quantities {
  real mu_eta;
  real<lower=0> mu_theta;
  real<lower=0, upper=max_lambda> mu_lambda;

  real log_lik[nS];
  int<lower=-1,upper=1> y_pred[nS, nT];

  mu_eta    = mu_p[1];
  mu_theta  = exp(mu_p[2]);
  mu_lambda = Phi_approx(mu_p[3]) * max_lambda;
  y_pred    = rep_array(-1, nS, nT);

  { // local block
    for (s in 1:nS) {
      vector[nT] delta;

      log_lik[s] = 0;
      delta = (intensity - eta[s]) / theta[s];

      for (t in 1:nT) {
        log_lik[s]  = log_lik[s] + bernoulli_lpmf( resp[s,t] | (1-lambda[s]) * Phi(delta[t]) + lambda[s]/2 );
        y_pred[s,t] = bernoulli_rng( (1-lambda[s]) * Phi(delta[t]) + lambda[s]/2 );       
      }      
    }
  }

}
