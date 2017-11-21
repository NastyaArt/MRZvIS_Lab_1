#include <QCoreApplication>

#include <iostream>
#include <CImg.h>
#include <vector>
#include <armadillo>

#define ARMA_DONT_USE_WRAPPER

#define C_MAX 255

#include "rectanglemodel.h"
#include "imagemodel.h"

using namespace std;
using namespace cimg_library;
using namespace arma;


int main()
{
  //  Q_INIT_RESOURCE(images);
    //CImg<unsigned char> src("user2.bmp");
    //mat W;

    ImageModel model("images/image2.bmp");
    model.run();
    model.createOutputImage();
    return 0;
}

/*
void initConditions();
void createOutputImage();
int convertColorToOutput(double x);
int n, m, p;
int N;
mat Y(1, p);
mat X_(1, N);
mat deltaX(1, N);
double e, alfa, alfa_;
int imageWidth;
int imageHeight;
int L;
vector<mat> imageList;
mat W; // (N, p); // матрица весов на 1 слое
mat W_(p, N); // матрица весов на втором слое

int main()
{
    CImg <unsigned char> image("images/image4.bmp");
    imageWidth = image.width();
    imageHeight = image.height();
    cout << "imageWidth = " << imageWidth << " " << "imageHeight = " << imageHeight << " " << endl;
    initConditions();

    N = n*m * 3;
    vector<mat> imageList;

    int k;
    for (int indexW = 0; indexW < imageWidth; indexW += m){
        for (int indexH = 0; indexH < imageHeight; indexH += n){
            mat X0 = mat(1, N);
            k = 0;
            for (int i = indexW; i < indexW + m; i++){
                for (int j = indexH; j < indexH + n; j++) {
                    if (i < imageWidth && j < imageHeight){
                        X0(0, k) = (((2.0 * ((int)image(i, j, 0, 0) / C_MAX)) - 1));
                        X0(0, k + 1) = (((2.0 * ((int)image(i, j, 0, 1) / C_MAX)) - 1));
                        X0(0, k + 2) = (((2.0 * ((int)image(i, j, 0, 2) / C_MAX)) - 1));
                        k = k + 3;
                    }
                    else
                    {
                        X0(0, k) = -1;
                        X0(0, k + 1) = -1;
                        X0(0, k + 2) = -1;
                        k = k + 3;
                    }
                }
            }
            imageList.push_back(X0);
        }
    }
    L = imageList.size();
    cout << "L=" << L << endl;
    W = randu <mat>(N, p); // матрица весов на первом слое
    srand(time(0));
    for (int i = 0; i < N; i++){
        for (int j = 0; j < p; j++){
            W(i, j) = (((double)rand() / RAND_MAX) * 2 - 1)*0.1;;
        }
    }
    W_ = W.t();

    int iterations = 0;
    double Eq;
    double E; //суммарная ошибка
    do {
        E = 0;
        for (int i = 0; i < L; i++){
            mat X = imageList[i];
            Y = X * W;
            X_ = Y * W_;
            deltaX = X_ - X;
            W = W - (alfa * X.t() * deltaX * W_.t());
            W_ = W_ - (alfa * Y.t() * deltaX);
        }

        for (int j = 0; j < L; j++){
            mat X = imageList[j];
            Y = X * W;
            X_ = Y * W_;
            deltaX = X_ - X;
            Eq = 0;
            for (int i = 0; i < N; i++){
                Eq += pow(deltaX(0, i), 2);
            }
            E += Eq;
        }
        cout << "iteration = " << iterations << "; " << "E = " << E << endl;
        iterations++;

    } while (E > e);
    int l = 0;
    CImg <float> image1(imageWidth, imageHeight, 1, 3, 0);
    for (int indexW = 0; indexW < imageWidth; indexW += m){
        for (int indexH = 0; indexH < imageHeight; indexH += n){
            mat X = imageList[l];
            Y = X * W;
            X_ = Y * W_;
            int k = 0;
            double color[3];
            for (int i = indexW; i < indexW + m; i++){
                for (int j = indexH; j < indexH + n; j++) {
                    color[0] = convertColorToOutput(X(0, k++));
                    color[1] = convertColorToOutput(X(0, k++));
                    color[2] = convertColorToOutput(X(0, k++));
                    if (i < imageWidth && j < imageHeight){
                        image1.draw_point(i, j, color);
                    }

                }
            }
            l++;
        }
    }
    image.save("result1.bmp");
    double Z = (N*L) / ((N + L)*p + 2);
    printf("%f", Z);
    cout << "Z = " << Z << "; \n iterations = " << iterations << "; \n";
    cout << "E = " << E << endl;
    return 0;
}

void createOutputImage()
{
    int l = 0;
    CImg <float> image(imageWidth, imageHeight, 1, 3, 0);
    for (int imageW = 0; imageW < imageWidth; imageW++){
        for (int imageH = 0; imageH < imageHeight; imageH++){
            mat X = imageList[l];
            Y = X * W;
            X_ = Y * W_;
            int k = 0;
            double color[3];
            for (int i = imageW; i < imageW + m; i++){
                for (int j = imageH; j < imageH + n; j++){
                    color[0] = convertColorToOutput(X(0, k++));
                    color[1] = convertColorToOutput(X(0, k++));
                    color[2] = convertColorToOutput(X(0, k++));
                    if (i < imageWidth && j < imageHeight){
                        image.draw_point(i, j, color);
                    }

                }
            }
            l++;
        }
    }
    image.save("result.bmp");
}
int convertColorToOutput(double x){
    double u;
    u = C_MAX*(x + 1) / 2;
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    return (int)u;
}

void initConditions(){
    cout << "Enter small image height(n) \n";
    cin >> n;
    cout << "Enter small image width(m) \n";
    cin >> m;
    cout << "Enter number of second layer neurons(p) \n";
    cin >> p;
    cout << "Enter max error(e). 0 < e < 0.1*p \n";
    cin >> e;
    cout << "Enter learning rate (alfa). 0< alfa<=0.01 (alfa <= e) \n";
    cin >> alfa;
}
*/
