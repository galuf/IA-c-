#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;


int main() {

  MatrixXf x1(15,2);  // LLenar matrix mediante funcion de lectura de csv (Feature)
  MatrixXf y(15,1); // LLenar con los elementos target del Data Set (Target)

  x1(0,0) = 15; x1(0,1)=70;   
  x1(1,0) = 16; x1(1,1)=65;   
  x1(2,0) = 24; x1(2,1)=71;   
  x1(3,0) = 13; x1(3,1)=64;   
  x1(4,0) = 21; x1(4,1)=84;   
  x1(5,0) = 16; x1(5,1)=86;   
  x1(6,0) = 22; x1(6,1)=72;   
  x1(7,0) = 18; x1(7,1)=84;   
  x1(8,0) = 20; x1(8,1)=71;   
  x1(9,0) = 16; x1(9,1)=75;   
  x1(10,0) = 28; x1(10,1)=84; 
  x1(11,0) = 27; x1(11,1)=79; 
  x1(12,0) = 13; x1(12,1)=80; 
  x1(13,0) = 22; x1(13,1)=76; 
  x1(14,0) = 23; x1(14,1)=88; 

  y(0,0)=156;
  y(1,0)=157;
  y(2,0)=177;
  y(3,0)=145;
  y(4,0)=197;
  y(5,0)=184;
  y(6,0)=172;
  y(7,0)=187;
  y(8,0)=157;
  y(9,0)=169;
  y(10,0)=200;
  y(11,0)=193;
  y(12,0)=167;
  y(13,0)=170;
  y(14,0)=192;

  MatrixXf x0 = MatrixXf::Ones(15, 1);
  MatrixXf x(15, 3);
  x << x0, x1; // Agregamos una columna de 1 a la matriz x1

  // train estimator
  Eigen::LeastSquaresConjugateGradient<Eigen::MatrixXf> gd;
  gd.setMaxIterations(1000); // Parametros a modificar para mejorar el resultado
  gd.setTolerance(0.0001f); // Parametros a modificar para mejorar el resultado
  gd.compute(x);
  Eigen::VectorXf b = gd.solve(y); // b -> son los parametros (Thetas) entrenados
  cout << "Estimated parameters vector : \n" << b <<endl;

  // Predecir
  Eigen::MatrixXf new_x(4,3); 
  // new_x es la matriz de prueba. Puedes llenar esta matriz con data real. 
  new_x <<1,15, 70,
          1,16, 65,
          1,42, 71,
          1,13, 64;
  //cout<<new_x<<endl;
  auto new_y = new_x.array().rowwise() * b.transpose().array();
  MatrixXf y_pred = new_y;
  MatrixXf pass_y = MatrixXf::Ones(new_y.cols(), 1);
  cout<<"Y predecido: \n"<<y_pred*pass_y<<endl; // Este valor es el predecido por tu modelo.

  return 0;
};