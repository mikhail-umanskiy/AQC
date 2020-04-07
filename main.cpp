#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/unsupported/Eigen/KroneckerProduct"
#include <time.h>
#include <math.h>
#include <complex>
using namespace Eigen;
#define SPIN_NUM 3 // number of particles
#define STEP_NUM 10000000 // number of steps for adiabatic computation
const int N = SPIN_NUM;
const double dt = 0.001;
const double q = 0.001; //dE/dt
double E = STEP_NUM*q*dt;

const int D = pow(2, N); // Hamiltonian will be a DxD matrix
Matrix2i Pauli_x, Pauli_z, I = Matrix2i::Identity();

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
  MatrixXd  H   =   H0;
  MatrixXcd k1,k2,k3,k4; 
  int percent = 0; 
  for (int i = 0; i < STEP_NUM; i++) {
  if (i == STEP_NUM*percent/100)
  {
    printf("%d%%\n", percent);
    percent++;
  }
  k1 = H*psi*(-std::complex<double>(0,1));
  
  H -= q*dt*_Sx*0.5;
  k2 = H*(psi+dt/2*k1)*(-std::complex<double>(0,1));
  k3 = H*(psi+dt/2*k2)*(-std::complex<double>(0,1));
  H -= q*dt*_Sx*0.5;
  k4 = H*(psi+dt*k3)*(-std::complex<double>(0,1));

  psi += dt/6*(k1+2*k2+2*k3+k4);
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
