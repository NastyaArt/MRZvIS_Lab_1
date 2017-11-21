#include <QCoreApplication>

#include <iostream>
#include <CImg.h>
#include <vector>
#include <armadillo>

#define ARMA_DONT_USE_WRAPPER

#define C_MAX 255
/*
#include "rectanglemodel.h"
#include "imagemodel.h"
*/
using namespace std;
using namespace cimg_library;
using namespace arma;

/*
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

*/
void initConditions();
int convertColorToOutput(double x);

int n; //ширина прямоугольника
int m; //высота прямоугольника
int p; //число нейронов второго слоя
int N; //длинна эталонного образа L
mat Y(1, p);    //Y(i) = X(i)*W
mat X_(1, N);   //X_(i) = Y(i)*W_
mat deltaX(1, N);   //∆X(i) = X_(i) – X(i)
double e; //максимальная допустимая ошибка
double alfa;  //коэффициент обучения для X
double alfa_;  //коэффициент обучения для Y
int imageWidth; //ширина исходного изображения
int imageHeight;  //высота исходного изображения
int L; //количество прямоугольников
vector<mat> imageList;  //список прямоугольников
mat W; // (N, p)  матрица весов на 1 слое
mat W_(p, N); // матрица весов на втором слое


void initConditions(){
    cout << "Enter rectangle's height(n) \n";  //высота прямоугольника
    cin >> n;
    cout << "Enter rectangle's width(m) \n";  //ширина прямоугольника
    cin >> m;
    cout << "Enter number of neurons on the second layer (p) \n";  //количество нейронов на втором слое
    cin >> p;
    cout << "Enter max error(e). 0 < e < 0.1*p \n";  //максимально допустимая ошибка
    cin >> e;
    cout << "Enter learning coefficient (alfa). 0< alfa<=0.01 (alfa <= e) \n";   //коэффициент обучения
    cin >> alfa;
    alfa_=alfa;

}

int convertColorToOutput(double x){
    double u;
    u = C_MAX*(x + 1) / 2;
    if (u < 0) u = 0;
    if (u > 255) u = 255;
    return (int)u;
}


//нормализация матрицы
void normalizeMatrix(mat matrix){
    for (unsigned int i = 0; i < matrix.n_cols; i++) {		//перебор матрицы по колонкам
        double sum = 0;
        for (unsigned int j = 0; j < matrix.n_rows; j++) {	//перебор матрицы по строкам
            sum += pow(matrix(j, i), 2);					//вычисление суммы квадратов значений элементов матрицы
        }
        sum = sqrt(sum);									//корень от суммы квадратов
        for (unsigned int j = 0; j < matrix.n_rows; j++) {
            matrix(j, i) = matrix(j, i) / sum;				//делим значение элемента на сумму - нормализуем значение
        }
    }
}

//нормализация весов на первом и втором слое
void normalizeMatrixs(){
    normalizeMatrix(W);		//нормализуем веса первого слоя
    normalizeMatrix(W_);	//нормализуем веса второго слоя
}

//расчет адаптивного шага обучения
double adaptiveLearningStep(mat matrix){  //тоже нестыковка с формулой
  /*  int FACTOR = 10;  //???
    mat temp = (matrix * matrix.t());
    return 1.0 / (temp(0,0) + FACTOR);
*/
    double a = 1.0;
    for (int i = 0; i < matrix.n_cols; ++i) {
        a += matrix(0, i) * matrix(0, i);
    }
    a = 1.0 / a;  //adapt step
    return a;

}

void start(const char * file, const char *outputFile)
{
    CImg <unsigned char> image(file);
    imageWidth = image.width();
    imageHeight = image.height();
   // cout << "imageWidth = " << imageWidth << " " << "imageHeight = " << imageHeight << " " << endl;
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
                        X0(0, k) = (((2.0 * image(i, j, 0, 0) / C_MAX) - 1));
                        X0(0, k + 1) = (((2.0 * image(i, j, 0, 1) / C_MAX) - 1));
                        X0(0, k + 2) = (((2.0 * image(i, j, 0, 2) / C_MAX) - 1));
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

    srand(time(NULL));  //генератор псевдо-случайных чисел
    W = randu <mat>(N, p); // матрица весов на первом слое
    for (int i = 0; i < N; i++){
        for (int j = 0; j < p; j++){
            W(i, j) = (((double)rand() / RAND_MAX) * 2 - 1)*0.1;;
        }
    }
    W_ = W.t();

    normalizeMatrixs();
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
            alfa_ = adaptiveLearningStep(Y);
            alfa = adaptiveLearningStep(X);
            W = W - (alfa * X.t() * deltaX * W_.t());
            W_ = W_ - (alfa_ * Y.t() * deltaX);
            normalizeMatrixs();
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
                    color[0] = convertColorToOutput(X_(0, k++));
                    color[1] = convertColorToOutput(X_(0, k++));
                    color[2] = convertColorToOutput(X_(0, k++));
                    if (i < imageWidth && j < imageHeight){
                        image1.draw_point(i, j, color);
                    }

                }
            }
            l++;
        }
    }
    image1.save(outputFile);
    double Z = (N*L) / ((N + L)*p + 2);
    cout << "Z = " << Z << "; \nIterations = " << iterations << "; \n";
    cout << "E = " << E << endl;

}

int main()
{
    start("images/image2.bmp", "result_images/image2_result_test4_with_norm_and_adept.bmp");
    return 0;
}


