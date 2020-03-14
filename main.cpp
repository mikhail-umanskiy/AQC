#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/unsupported/Eigen/KroneckerProduct"
#include <time.h>
#include <math.h>
using namespace Eigen;

#define N 2 // number of particles
const float E = 100.0;

const int D = pow(2, N); // Hamiltonian will be a DxD matrix
Matrix2i Pauli_x, Pauli_z, I = Matrix2i::Identity();

int main()
{
  srand(time(NULL));
  Pauli_x << 0, 1, // Pauli matrices
             1, 0;
  Pauli_z << 1, 0,
             0, -1;

  MatrixXf J = MatrixXf::Random(N,N), tmp1(D,D); 
  MatrixXf H1(D, D), H0(D,D);
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
      H1 += J(i,j) * (Sz[i] * Sz[j]).cast<float>(); 
  }
  H1 /= D;

  H0 = H1; 
  for (int i = 0; i < N; i++) H0 += E / 2 * Sx[i].cast<float>(); // calculating H0

  std::cout << "J:" << std::endl << J << std::endl << std::endl;
  std::cout << "E:" << std::endl << E << std::endl << std::endl;
  std::cout << "H1:" << std::endl << H1 << std::endl << std::endl;
  std::cout << "H0:" << std::endl << H0 << std::endl << std::endl;
  std::cout << std::endl << "H0 eigenvalues: " << std::endl << H0.eigenvalues() << std::endl; 

  return 0;
} 
