#define BOOST_TEST_MODULE kf 
#include <time.h>
#include <boost/test/included/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "kf.cpp"

BOOST_AUTO_TEST_CASE( speed_test )
{
  const unsigned cnt = 100000;
  
  boost::mt19937 randgen(0); 
  boost::normal_distribution<float> n(0.0, 1.0);
  boost::variate_generator< boost::mt19937 &, 
    boost::normal_distribution<float> > Z(randgen, n);
  
  double xs[cnt][10];
  double noise[cnt];
  for (unsigned t = 0; t < cnt; ++ t) {
    for (unsigned i = 0; i < 10; ++ i) 
      xs[t][i] = Z();
    noise[t] = Z();
  }
  KF<1> kf1;
  KF<2> kf2;
  KF<10> kf10;
  vector<double> x1(1);
  vector<double> x2(2);
  vector<double> x10(10);
  double y = 0.0;
  time_t begin, end;


  time(&begin);
  for (unsigned t = 0; t < cnt; ++ t) {
    x1(0) = xs[t][0];
    kf1(y, x1);
    y = x1(0) + noise[t];
  }
  time(&end);
  std::cout << "Univariate runtime = " << difftime(end, begin) << std::endl;

  time(&begin);
  for (unsigned t = 0; t < cnt; ++ t) {
    x2(0) = xs[t][0];
    x2(1) = xs[t][1];
    kf2(y, x2);
    y = x2(0) + x2(1) + noise[t];
  }
  time(&end);
  std::cout << "Bivariate runtime = " << difftime(end, begin) << std::endl;


  time(&begin);
  for (unsigned t = 0; t < cnt; ++ t) {
    for (unsigned i = 0; i < 10; ++ i) 
      x10(i) = xs[t][i];
    kf10(y, x10);
    for (unsigned i = 0; i < 10; ++ i) 
      y += x10(i);
    y += noise[t];
  }
  time(&end);
  std::cout << "10-variate runtime = " << difftime(end, begin) << std::endl;
}

