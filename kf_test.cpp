#define BOOST_TEST_MODULE kf 
#include <math.h>
#include <boost/test/included/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "kf.cpp"

#define TOL 0.000001

#define PROBE \
std::cout << \
" beta=" << kf.beta << \
" k=" << kf.k << \
" Delta_root_inv=" << kf.Delta_root_inv << \
" yPred=" << kf.yPred << \
" m=" << kf.m << \
" P=" << kf.P << \
" R=" << kf.R << \
" K=" << kf.K << \
" Q=" << kf.Q << \
" S=" << kf.S << \
std::endl;

BOOST_AUTO_TEST_CASE( initialization )
{
  KF<> kf;

  BOOST_CHECK_LE(kf.beta, 1.0);
  BOOST_CHECK_GE(kf.beta, 0.0);
  BOOST_CHECK_GE(kf.Delta_root_inv(0,0), 1.0);

  KF<1> kf1;
  BOOST_CHECK_GE(kf1.Delta_root_inv(0,0), 1.0);

  KF<2> kf2;
  BOOST_CHECK_GE(kf2.Delta_root_inv(0,0), 1.0);
  BOOST_CHECK_GE(kf2.Delta_root_inv(1,1), 1.0);

  KF<1> kf3(.5, vector<double> (1, .25));
  BOOST_CHECK_EQUAL(kf3.beta, .5);
  BOOST_CHECK_CLOSE(kf3.Delta_root_inv(0,0), 2.0, TOL);
}

BOOST_AUTO_TEST_CASE( coefficient_increases )
{
  KF<> kf;

  vector<double> x(1);
  x(0) = 1.0;

  double m[3]; 
  kf(0.0, x);
  kf(0.0, x);

  m[1] = kf.m(0);
  kf(1.0, x);
  m[2] = kf.m(0);
  for (unsigned t = 0; t < 20; ++ t) {
    kf(1.0, x); 
    m[0] = m[1]; 
    m[1] = m[2]; 
    m[2] = kf.m(0);
    // Coefficent must rise exponentially
    BOOST_CHECK_GT(m[1], m[0]);
    BOOST_CHECK_GT(m[2], m[1]);
    BOOST_CHECK_GT(m[1]-m[0], m[2]-m[1]);
  }
}

BOOST_AUTO_TEST_CASE( coefficient_decreases )
{
  KF<> kf;

  vector<double> x(1);
  x(0) = 1.0;

  double m[3]; 
  kf(1.0, x);
  kf(1.0, x);

  m[1] = kf.m(0);
  kf(0.0, x);
  m[2] = kf.m(0);
  for (unsigned t = 0; t < 100; ++ t) {
    kf(0.0, x); 
    m[0] = m[1]; 
    m[1] = m[2]; 
    m[2] = kf.m(0);
    // Coefficent must rise exponentially
    BOOST_CHECK_LT(m[1], m[0]);
    BOOST_CHECK_LT(m[2], m[1]);
    BOOST_CHECK_LT(m[1]-m[0], m[2]-m[1]);
  }
}

BOOST_AUTO_TEST_CASE( correct_after_long_time )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;
  kf(0.0, x); // scramble non-informative prior
  kf(1.0, x); // scramble non-informative prior

  for (unsigned t = 0; t < 1000; ++ t) {
    kf(0.3, x);
  }
  double ypred = kf(0.3, x);
  BOOST_CHECK_CLOSE_FRACTION(kf.m(0), 0.3, TOL);
  BOOST_CHECK_CLOSE_FRACTION(ypred, 0.3, TOL);
}

BOOST_AUTO_TEST_CASE( correct_after_long_time_in_noise )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> noise(0.0, 0.5);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > nD(randgen, noise);

  for (unsigned t = 0; t < 1000; ++ t) 
    kf(0.3 + nD(), x);
  
  double ypred = kf(0.3, x);
  BOOST_CHECK_CLOSE_FRACTION(kf.m(0), 0.3, .05); // 5% accuracy
  BOOST_CHECK_CLOSE_FRACTION(ypred, 0.3, .05);
}

BOOST_AUTO_TEST_CASE( better_than_static_on_dynamic_toy_data )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;

  double error; 

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> noise(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > nD(randgen, noise);

  for (unsigned i = 0; i < 5; ++ i) {
    for (unsigned t = 0; t < 100; ++ t) 
      error += fabs(kf(0.0 + nD(), x) - 0.0);
    for (unsigned t = 0; t < 100; ++ t) 
      error += fabs(kf(1.0 + nD(), x) - 1.0);
  }

  double error_nondynamic = 1000*.5; 

  BOOST_CHECK_LT(error, error_nondynamic); 
}

BOOST_AUTO_TEST_CASE( coefficient_is_not_sticky )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;

  double error; 

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> noise(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > nD(randgen, noise);

  double sm[5];
  double lg[5];
  for (unsigned i = 0; i < 10; ++ i) {
    for (unsigned t = 0; t < 200; ++ t) 
      error += fabs(kf(0.0 + nD(), x) - 0.0);
    if (i >= 5) sm[i-5] = kf.m(0);
    for (unsigned t = 0; t < 200; ++ t) 
      error += fabs(kf(1.0 + nD(), x) - 1.0);
    if (i >= 5) lg[i-5] = kf.m(0);
  }

  // make sure coefficient doesn't stagnate near the mean
  // it must stay flexible and keep adapting at a similar rate
  // allow some burn-in by ignoring first 5 values
  BOOST_CHECK(!(sm[0] < sm[1] && sm[1] < sm[2] && sm[2] < sm[3] && sm[3] < sm[4])); 
  BOOST_CHECK(!(lg[0] > sm[1] && sm[1] > sm[2] && sm[2] > sm[3] && sm[3] > sm[4])); 
}

BOOST_AUTO_TEST_CASE( confidence_intervals_correct )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;
  double noiseSD = 0.5;

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> noise(0.0, noiseSD);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > nD(randgen, noise);

  for (unsigned t = 0; t < 1000; ++ t)
    kf(0.3 + nD(), x);
  
  double confidence;
  kf(0.3, x, confidence);
  BOOST_CHECK_CLOSE_FRACTION(sqrt(confidence), 0.5, .05); // 5% accuracy
}

BOOST_AUTO_TEST_CASE( confidence_interval_dynamic )
{
  KF<> kf;
  vector<double> x(1);
  x(0) = 1.0;
  double noiseSD = 0.5;

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> noise(0.0, noiseSD);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > nD(randgen, noise);

  for (unsigned t = 0; t < 1000; ++ t)
    kf(0.3 + nD(), x);
  double confidence;
  kf(0.3, x, confidence);
  BOOST_CHECK_CLOSE_FRACTION(sqrt(confidence), noiseSD, .05); // 5% accuracy

  for (unsigned t = 0; t < 1000; ++ t)
    kf(0.3 + 2.0*nD(), x);
  kf(0.3, x, confidence);
  BOOST_CHECK_CLOSE_FRACTION(sqrt(confidence), 2.0*noiseSD, .05); 

  for (unsigned t = 0; t < 1000; ++ t)
    kf(0.3 + 5.0*nD(), x);
  kf(0.3, x, confidence);
  BOOST_CHECK_CLOSE_FRACTION(sqrt(confidence), 5.0*noiseSD, .05); 

  for (unsigned t = 0; t < 1000; ++ t)
    kf(0.3 + nD(), x);
  kf(0.3, x, confidence);
  BOOST_CHECK_CLOSE_FRACTION(sqrt(confidence), noiseSD, .05); 
}

BOOST_AUTO_TEST_CASE( bivariate_coefficients_correct_after_long_time )
{
  KF<2> kf(.999, vector<double>(2, .999));
  vector<double> x(2);
  vector<double> m(2); 
  m(0) = 0.5;
  m(1) = 1.0;
    
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  double y = 0.0;
  for (unsigned t = 0; t < 2000; ++ t) {
    x(0) = Z(); 
    x(1) = Z();
    kf(y, x);
    y = inner_prod(x, m) + Z();
  }
  
  x(0) = 2.0; 
  x(1) = 3.0;
  double ytrue = inner_prod(x,m);
  double ypred = kf(1.5, x);
  BOOST_CHECK_CLOSE_FRACTION(kf.m(0), m(0), .05); // 5% accuracy
  BOOST_CHECK_CLOSE_FRACTION(kf.m(1), m(1), .05); 
  BOOST_CHECK_CLOSE_FRACTION(ypred, ytrue, .05);
}

BOOST_AUTO_TEST_CASE( bivariate_better_than_static_on_dynamic_toy_data )
{
  KF<2> kf; 
  vector<double> x(2);
  vector<double> m(2); 
  double y;
    
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  double error; 

  for (unsigned i = 0; i < 5; ++ i) {
    m(0) = 1.0; 
    m(1) = 1.0;
    for (unsigned t = 0; t < 200; ++ t) {
      x(0) = Z();
      x(1) = Z();
      error += fabs(kf(y, x) - inner_prod(x, m));
      y = inner_prod(x, m) + Z();
    }
    m(0) = 0.0; 
    m(1) = 0.0;
    for (unsigned t = 0; t < 200; ++ t) {
      x(0) = Z();
      x(1) = Z();
      error += fabs(kf(y, x) - inner_prod(x, m));
      y = inner_prod(x, m) + Z();
    }
  }

  double error_nondynamic = 2000*.5; 

  BOOST_CHECK_LT(error, error_nondynamic); 
}

BOOST_AUTO_TEST_CASE( bivariate_coefficients_correlation_equals_one )
{
  KF<2> kf(.999, vector<double>(2, .999));
  vector<double> x(2);
  vector<double> m(2); 
  m(0) = 0.5;
  m(1) = 1.0;
    
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  double y = 0.0;
  for (unsigned t = 0; t < 1000; ++ t) {
    x(0) = Z(); 
    x(1) = x(0);
    kf(y, x);
    y = inner_prod(x, m) + Z();
  }
  
  x(0) = 0.5; 
  x(1) = 0.5;
  double ytrue = inner_prod(x,m);
  double ypred = kf(0.0, x);
  BOOST_CHECK_CLOSE_FRACTION(ypred, ytrue, .05);
}

BOOST_AUTO_TEST_CASE( correct_correlation )
{
  KF<2> kf(.999, vector<double>(2, .999));
  vector<double> x(2);
  vector<double> m(2); 
  m(0) = 0.5;
  m(1) = 1.0;
  double rho = 0.5;

  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  double y = 0.0;
  for (unsigned t = 0; t < 5000; ++ t) {
    x(0) = Z();
    x(1) = rho*x(0) + sqrt(1-rho*rho)*Z();
    kf(y, x);
    y = m(0)*x(0) + m(1)*x(1) + 0.5*Z();
  }
  
  x(0) = 2.0;
  x(1) = 3.0;
  double ytrue = inner_prod(x,m);
  double ypred = kf(1.5, x);
  BOOST_CHECK_CLOSE_FRACTION(kf.m(0), m(0), .05); // 5% accuracy
  BOOST_CHECK_CLOSE_FRACTION(kf.m(1), m(1), .05); 
  BOOST_CHECK_CLOSE_FRACTION(ypred, ytrue, .05);
  BOOST_CHECK_CLOSE_FRACTION(fabs(rho*(kf.P(0,0)+kf.P(1,1))/2.0), fabs(kf.P(1,0)), .05);
}

BOOST_AUTO_TEST_CASE( coefficients_couple_then_decouple )
{
  KF<2> kf(.99, vector<double>(2, .99));
  vector<double> x(2);
  double y;
  double rho;
  vector<double> m(2); 
  m(0) = 0.5;
  m(1) = 1.0;
    
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  rho = 0.5;
  for (unsigned t = 0; t < 500; ++ t) {
    x(0) = Z();
    x(1) = rho*x(0) + sqrt(1-rho*rho)*Z();
    kf(y, x);
    y = m(0)*x(0) + m(1)*x(1) + 0.5*Z();
  }
  BOOST_CHECK_CLOSE_FRACTION(fabs(rho*(kf.P(0,0)+kf.P(1,1))/2.0), fabs(kf.P(1,0)), .05);
  
  // decouple
  rho = 0.0;
  for (unsigned t = 0; t < 500; ++ t) {
    x(0) = Z();
    x(1) = rho*x(0) + sqrt(1-rho*rho)*Z();
    kf(y, x);
    y = m(0)*x(0) + m(1)*x(1) + 0.5*Z();
  }
  BOOST_CHECK_SMALL(kf.P(1,0), .05);

  // strongly recouple
  rho = 0.9;
  for (unsigned t = 0; t < 500; ++ t) {
    x(0) = Z();
    x(1) = rho*x(0) + sqrt(1-rho*rho)*Z();
    kf(y, x);
    y = m(0)*x(0) + m(1)*x(1) + 0.5*Z();
  }
  BOOST_CHECK_CLOSE_FRACTION(fabs(rho*(kf.P(0,0)+kf.P(1,1))/2.0), fabs(kf.P(1,0)), .05);

  // back to start
  rho = 0.5;
  for (unsigned t = 0; t < 500; ++ t) {
    x(0) = Z();
    x(1) = rho*x(0) + sqrt(1-rho*rho)*Z();
    kf(y, x);
    y = m(0)*x(0) + m(1)*x(1) + 0.5*Z();
  }
  BOOST_CHECK_CLOSE_FRACTION(fabs(rho*(kf.P(0,0)+kf.P(1,1))/2.0), fabs(kf.P(1,0)), .05);
}

BOOST_AUTO_TEST_CASE( ten_coefficients )
{
  KF<10> kf(.9999, vector<double>(10, .9999));
  vector<double> x(10);
  vector<double> m(10); 
  double y;
    
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);

  for (unsigned i = 0; i < 10; ++ i) {
    m(i) = Z(); 
  }

  for (unsigned t = 0; t < 5000; ++ t) {
     for (unsigned i = 0; i < 10; ++ i) {
       x(i) = Z(); 
     } 
    kf(y, x);
    y = inner_prod(x, m) + Z();
  }
  
  for (unsigned i = 0; i < 10; ++ i) {
    x(i) = Z(); 
  } 

  double ytrue = inner_prod(x,m);
  double ypred = kf(0.0, x);
  BOOST_CHECK_CLOSE_FRACTION(ypred, ytrue, .05);
  for (unsigned i = 0; i < 10; ++ i) {
    BOOST_CHECK_CLOSE_FRACTION(kf.m(i), m(i), .05); 
  } 
}
