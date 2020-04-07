#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/unsupported/Eigen/KroneckerProduct"
#include <time.h>
#include <math.h>
#include <complex>
using namespace Eigen;
#define SPIN_NUM 3 // number of particles
const int N = SPIN_NUM;
const double dt = 0.001;
const double q = 0.001;
const double E0 = 5;
const int STEP_NUM = (int)(E0/(q*dt));
double E = E0;
const int D = pow(2, N); // Hamiltonian will be a DxD matrix

Matrix2i Pauli_x, Pauli_z, I = Matrix2i::Identity();
const std::complex<double> _i(0,1);
const double _s21 = sqrt(21);

int main()
{
  srand(time(NULL));
  Pauli_x << 0, 1, // Pauli matrices
             1, 0;
  Pauli_z << 1, 0,
             0, -1;

  MatrixXd J = MatrixXd::Random(N,N), tmp1(D,D); 
  MatrixXd H1(D, D), H0(D,D);
  MatrixXi Sz[N], Sx[N], tmp[N+1];
  for (int i = 0; i < N; i++) Sz[i] = MatrixXi(D,D); 
  
  for (int i = 0; i < N; i++) // calculating Sz
  {
    for (int j = 0; j < N+1; j++) tmp[j] = MatrixXi::Identity(pow(2,j), pow(2,j)); 
    tmp[i+1] = kroneckerProduct(tmp[i], Pauli_z);
    for (int j = i+1; j < N; j++) tmp[j+1] = kroneckerProduct(tmp[j], I);
    Sz[i] = tmp[N];
  }

  for (int i = 0; i < N; i++) // calculating Sx
  {    
    for (int j = 0; j < N+1; j++) tmp[j] = MatrixXi::Identity(pow(2,j), pow(2,j)); 
    tmp[i+1] = kroneckerProduct(tmp[i], Pauli_x);
    for (int j = i+1; j < N; j++) tmp[j+1] = kroneckerProduct(tmp[j], I);
    Sx[i] = tmp[N];
  }
  
  for (int i = 0; i < N; i++) // calculating H1
  {
    for(int j = 0; j < N; j++)
      H1 += J(i,j) * (Sz[i] * Sz[j]).cast<double>(); 
  }
  H1 /= D;

  H0 = H1; 
  MatrixXd _Sx = Sx[0].cast<double>();
  for (int i = 1; i < N; i++) _Sx += Sx[i].cast<double>(); // calculating _Sx - sum of Sx[i]
  _Sx *= 0.5;
  H0 += E * _Sx;

  EigenSolver<MatrixXd> solver_H0(H0); // solver for H0
  //std::cout << "J:" << std::endl << J << std::endl << std::endl;
  //std::cout << "E:" << std::endl << E << std::endl << std::endl;
  //std::cout << "H1:" << std::endl << H1 << std::endl << std::endl;
  std::cout << "H0:" << std::endl << H0 << std::endl;
  std::cout << std::endl << "H0 eigenvalues: " << std::endl << H0.eigenvalues() << std::endl;
  std::cout << std::endl << "H0 eigenvectors: " << std::endl << solver_H0.eigenvectors() << std::endl; 
  
  double min = 0, minnum = 0;
  for (int i = 0; i < D; i++) if( H0.eigenvalues()(i).real() < min) { min = H0.eigenvalues()(i).real(); minnum = i; }
  MatrixXcd psi0 = solver_H0.eigenvectors().col(minnum);
  std::cout << std::endl << "psi0:" << std::endl << psi0 << std::endl;
  MatrixXcd psi = psi0;
  MatrixXd  H   =   H0, dH=-q*dt*_Sx;
  MatrixXcd k1,k2,k3,k4,k5,k6,k7; 
  int percent = 0; 
  for (int i = 0; i < STEP_NUM; i++) {
    if (i == STEP_NUM*percent/100)
    {
      printf("%d%%\n", percent);
      percent++;
    }
    k1 = -_i*H*psi;
    k2 = -_i*(H + dH)*(psi + k1*dt);
    k3 = -_i*(H + dH/2)*(psi + (3*k1 + k2)*dt/8);
    k4 = -_i*(H + 2.0/3.0*dH)*(psi + (8*k1 + 2*k2 + 8*k3)*dt/27);
    k5 = -_i*(H + (7.0 - _s21)/14*dH)*(psi + (3*(3*_s21 - 7)*k1 - 8*(7 - _s21)*k2 + 48*(7 - _s21)*k3 - 3*(21 - _s21)*k4)*dt/392);
    k6 = -_i*(H + (7.0 + _s21)/14*dH)*(psi + (-5*(231 + 51*_s21)*k1 - 40*(7 + _s21)*k2 - 320*_s21*k3 + 3*(21 + 121*_s21)*k4 + 392*(6 + _s21)*k5)*dt/1960);
    k7 = -_i*(H + dH)*(psi + (15*(22+7*_s21)*k1 + 120*k2 + 40*(7*_s21 - 5)*k3 - 63*(3*_s21 - 2)*k4 - 14*(49 + 9*_s21)*k5 + 70*(7 - _s21)*k6)*dt/180);

    psi += dt/180*(9*k1 + 64*k3 + 49*k5 + 49*k6 + 9*k7);
    H   += dH;
  }
  EigenSolver<MatrixXd> solver_H1(H1); // solver for H0
  std::cout << std::endl << std::endl << "H1:" << std::endl << H1 << std::endl;
  std::cout << std::endl << std::endl << "H1 eigenvalues:" << std::endl << solver_H1.eigenvalues() << std::endl;
  std::cout << std::endl << std::endl << "H1 eigenvectors:" << std::endl << solver_H1.eigenvectors() << std::endl;
  //std::cout << std::endl << std::endl << std::endl << solver_H0.eigenvectors() << std::endl;
  std::cout << std::endl << std::endl << "psi:" << std::endl << psi << std::endl;
  std::cout << std::endl << std::endl << "psi1:" << std::endl << psi*exp(-std::complex<double>(0,1)*atan(psi(0).imag()/psi(0).real())) << std::endl;
  std::cout << std::endl << std::endl << "psi2:" << std::endl << psi*exp(-std::complex<double>(0,1)*atan(psi(1).imag()/psi(1).real())) << std::endl;
  std::cout << std::endl << std::endl << "psi3:" << std::endl << psi*exp(-std::complex<double>(0,1)*atan(psi(2).imag()/psi(2).real())) << std::endl;
  std::cout << std::endl << std::endl << "psi4:" << std::endl << psi*exp(-std::complex<double>(0,1)*atan(psi(3).imag()/psi(3).real())) << std::endl;
  //exp(-std::complex<double>(0,1)*atan(psi(0).imag()/psi(0).real()));
  //std::cout << std::endl << std::endl << std::endl << psi << std::endl;
  //psi *= exp(-std::complex<double>(0,1)*atan(psi(0).imag()/psi(0).real()));

  return 0;
} 
