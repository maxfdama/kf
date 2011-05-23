#include <iostream>
#include <math.h> // for sqrt
#include <float.h> // for DBL_MAX
/* Boost Installation on Ubuntu:
   $ sudo apt-get install libboost-dev libboost-doc */
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/numeric/ublas/banded.hpp> // for diagonal_matrix
#include <boost/numeric/ublas/vector.hpp>

using namespace boost::numeric::ublas;

template <std::size_t p = 1>
class KF 
{
public:
  /*
    Configuration:
      beta, discount factor
      unsigned integer template parameter specifies number of variables

    Generative Model:
      Sigma_{t+1} = Sigma_t "+ Wishart",  "+" => conv. with Sing MV Beta 
      m_{t+1} = m_t + Omega_t,  Omega_t ~ N(0, W_t, Sigma_t)
      y_t = x_t m_t + epsilon_t,  epsilon_t ~ N(0, Sigma_t)
  */
  KF(double beta = .99, const vector<double>& Delta = vector<double> (p, .99)) 
    : Delta_root_inv(p),
      m(p),      
      P(p, p),
      R(p, p),
      K(p)
  { 
    // Hyperparameters 
    this->beta = beta;
    double p_d = p;
    k = (beta*(1-p_d)+p_d)/(beta*(2-p_d)+p_d-1);
    for (std::size_t i = 0; i < p; ++i) 
      Delta_root_inv(i,i) = 1.0/sqrt(Delta(i));

    // Priors
    S = 1;
    P = 1000 * identity_matrix<double>(p);
    m = zero_vector<double>(p);

    // DBL_MAX, essentially infinity, is used as a flag to indicate
    // that the first usage of the algorithm should not call update
    yPred = DBL_MAX; 
  }

  double operator()(double y, const vector<double>& x);
  double operator()(double y, const vector<double>& x, double& yVar);
  void estimate(const vector<double>& x);
  bool not_first_run() const { return yPred != DBL_MAX; }
  void update(double y);
  double predict(const vector<double>& x) const { return inner_prod(x, m); }
  double confidence() const { return Q*(1-beta)*S/(k*(3*beta-2)); }

  double beta; // Coeffient rate of change adaptiveness 
  double k; // calculated from beta
  diagonal_matrix<double> Delta_root_inv; // Coefficient rate of change

  double yPred; // return value from previous timestep, for updating
  vector<double> Kprev; // K from previous timestep, for updating

  vector<double> m; // estimated regression coefficient vector
  matrix<double> P; // covariance matrix of (m_true - m)
  matrix<double> R; // covariance matrix of (m_true - m-1)
  vector<double> K; // kalman gain, optimal weight for new update
  double Q;         // forecast variance Var(y), scalar
  double S;         // Wishart mean covariance of Sigma => P
};

/* 
   Overloaded to ignore confidence interval
   See below
*/
template <std::size_t p>
double KF<p>::operator()(double y, const vector<double>& x)
{
  double ignore;
  return this->operator()(y, x, ignore);
}

/*
  Inputs:
  y from previous timestep, for updating
  x current observation
  yVar reference to second return value, predicted y variance

  Output:
  predicted y value
*/

template <std::size_t p>
double KF<p>::operator()(double y, const vector<double>& x, double& yVar)
{
  estimate(x);

  if (not_first_run())
    update(y);

  Kprev = K;
  yPred = predict(x); 
  yVar = confidence(); 
  
  return yPred;
}

template <std::size_t p>
void KF<p>::estimate(const vector<double>& x)
{
  // uBlas requires introducing the temporary type matrix<double>
  R = prod(Delta_root_inv, matrix<double>(prod(P, Delta_root_inv)));
  Q = inner_prod(prod(trans(x), R), x) + 1; 
  K = prod(R, x) / Q;
  P = R - outer_prod(K, K) * Q;
}

template <std::size_t p>
void KF<p>::update(double y) 
{
  double e = y - yPred;
  S = S / k + e * e / Q;
  m += Kprev * e;
}

