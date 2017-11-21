#include <iostream>
#include <CImg.h>
#include <vector>
#include <armadillo>
#include <cmath>
#include <float.h>

#include "imagemodel.h"
#include "rectanglemodel.h"

using namespace std;
using namespace cimg_library;
using namespace arma;

//констуктор - разбиение изображения на прямоугольники и заполнение матриц входных параметров
ImageModel::ImageModel(char const * patch){
    CImg<unsigned char> src(patch);		//загружаем изображение
    initANN();		//вводим входные параметры
    imageWidth = src.width();		//устанавливаем ширину
    imageHeight = src.height();		//и высоту изображения
    for (int indexW = 0; indexW < imageWidth; indexW +=m){		//цикл по ширине изображения
        for (int indexH = 0; indexH < imageHeight; indexH +=n){		//цикл по высоте изображения
            RectangleModel bufferRectangle(indexW,indexH);		//создаем прямоугольник
            for (int i = indexW; i < indexW + m; i++){		//перебираем пиксели, соответствующие прямоугольнику
                for (int j = indexH; j < indexH + n; j++) {
                    if (i < imageWidth && j < imageHeight){		//если изображение не закончилось
                        bufferRectangle.addElement(convertColor((int)src(i,j,0,0)));		//добавляем преобразованный пиксель в вектор прямоугольника по R
                        bufferRectangle.addElement(convertColor((int)src(i,j,0,1)));		//по G
                        bufferRectangle.addElement(convertColor((int)src(i,j,0,2)));		//по B
                    } else {						//если выходим за границы изображения, то заполняем цвета "пустыми"										
                        bufferRectangle.addElement(-1);
                        bufferRectangle.addElement(-1);
                        bufferRectangle.addElement(-1);
                    }
                }
            }
            bufferRectangle.createMatrixX(); //создаем матрицу из вектора, который заполнили выше
            rectangleModelList.push_back(bufferRectangle);  //добавляем прямоугольник в список
        }
    }
    L = rectangleModelList.size();  //устанавливаем количество прямоугольников
    nmRGB = n * m * RGB;		//и количество пикселей в прямоугольнике с учетом "растроения" цвета
    createWeightMatrix();		//заполненяем матриц весов псевдослучайными числами
}
//ввод входных параметров
void ImageModel::initANN(){
    cout << "Enter hight of rectangle(n):" << endl;
    cin >> n;
    cout << "Enter width of rectangle(m):" << endl;
    cin >> m;
    cout << "Enter number of neuron for second layer(p):" << endl;
    cin >> p;
    cout << "Enter error degree(e):" << endl;
    cin >> e;
    cout << "Enter step(a) (Enter 0 for adaptive learnin step):" << endl;
    cin >> a;
}
//запуск работы с изображением
void ImageModel::run(){
    double step;	//коэффициент обучения а   для обучения нейронов первого слоя 
    double step_;	//коэффициент обучения а'  для корректировки весов на втором слое 
    double E;		//общая ошибка
    normalizeMatrixs(); // uncomment if necessary normalization  //нормализация
    int iteration = 0;  //номер итерации
    do {		//выполняем цикл, пока общая ошибка не станет меньше максисмально допустимой
        E = 0;	//обнуляем ошибку для текущей итерации
        for (int index = 0; index < L; index++){	//перебираем все прямоугольники
            mat X = rectangleModelList[index].getX();	//получаем матрицу цветов текущего прямоугольника
            mat Y = X * W;	//вычисляем выходную матрицу после первого слоя
            mat X_ = Y * W_;	//вычисляем выходную матрицу из второго слоя
            mat deltaX = X_ - X;	//вычисляем разницу в значениях исходной матрицы цветов и выходной матрицы цветов
            if (a){	//если указали коэффициент обучения !=0
                step_ = step = a;	
            } else {	//иначе вычисляем адаптивный шаг обучения по формуле
                step_ = adaptiveLearningStep(Y);
                step = adaptiveLearningStep(X);
            }
            W = W - (step * X.t() * deltaX * W_.t()); 	//обучение нейронов первого слоя 
            W_ = W_ - (step_ * Y.t() * deltaX);		//корректировка весов на втором слое 
            normalizeMatrixs(); // uncomment if necessary normalization  //нормализация
        }
        // count error after correction		//вычисляем ошибку после обучения и корректировки 
        for (int index = 0; index < L; index++){	//перебираем все прямоугольники
            mat X = rectangleModelList[index].getX();	//получаем матрицу цветов текущего прямоугольника
            mat Y = X * W;	//вычисляем выходную матрицу после первого слоя
            mat X_ = Y * W_;	//вычисляем выходную матрицу из второго слоя	
            mat deltaX = X_ - X;	//вычисляем разницу в значениях исходной матрицы цветов и выходной матрицы цветов
            E += getErrorDegree(deltaX);	//прибавляем к общей ошибке суммарную ошибку для обучающей выборки 
        }
        iteration++;	//переходим к следующей итерации
        cout << "Iteration: " << iteration << " Error: " << E << endl;  //выводим номер итерации и общую ошибку
    } while (E > e);		//пока общая ошибка не станет меньше максисмально допустимой продолжаем цикл
    double z = (1.0 * n * m * RGB * L) / ((n * m * RGB + L) * p + 2);	//вычисляем коэффициент сжатия Z по формуле
    cout << "Z = " << z << endl;;  //и выводим его
}
//нормализация весов на первом и втором слое
void ImageModel::normalizeMatrixs(){
    normalizeMatrix(W);		//нормализуем веса первого слоя
    normalizeMatrix(W_);	//нормализуем веса второго слоя
}
//нормализация матрицы
void ImageModel::normalizeMatrix(mat matrix){
    for (unsigned int i = 0; i < matrix.n_cols; i++) {		//перебор матрицы по колонкам
        double sum = 0;
        for (unsigned int j = 0; j < matrix.n_rows; j++) {	//перебор матрицы по строкам
            sum += pow(matrix(j, i), 2);					//вычисление суммы квадратов значений элдементов матрицы
        }
        sum = sqrt(sum);									//корень от суммы квадратов
        for (unsigned int j = 0; j < matrix.n_rows; j++) {
            matrix(j, i) = matrix(j, i) / sum;				//делим значение элемента на сумму - нормализуем значение
        }
    }
}
//расчет адаптивного шага обучения
double ImageModel::adaptiveLearningStep(mat matrix){  //тоже нестыковка с формулой
  /*  int FACTOR = 10;  //???
    mat temp = (matrix * matrix.t());
    return 1.0 / (temp(0,0) + FACTOR);
*/
           double a = 1.0;
            for (int i = 0; i < nmRGB; ++i) {
                a += matrix(0, i) * matrix(0, i);
            }

            a = 1.0 / a;  //adapt step
            return a;

}
//формирование выходного изображения
void ImageModel::createOutputImage(){
    CImg<float> image(imageWidth,imageHeight,1,3,0);	//создаем изображение	
    float color[3];										//три компоненты цвета
    for (int index = 0; index < L; index++){		//перебираем все прямоугольники
        int startX = rectangleModelList[index].getStartX();		//получаем координаты прямоугольника - с каких пикселей он начинается
        int startY = rectangleModelList[index].getStartY();		
        mat X = rectangleModelList[index].getX();	//получаем матрицу цветов текущего прямоугольника
        mat Y = X * W;								//вычисляем выходную матрицу после первого слоя
        mat X_ = Y * W_;							//вычисляем выходную матрицу из второго слоя
        int pixel = 0;								//номер пикселя
        for (int i = startX; i < m + startX; i++) {			//перебираем пиксели по широте
            for (int j = startY; j < n + startY; j++) {			//перебираем пиксели по высоте
                color[0] = convertRGBToOutput(X_(0, pixel++));	//преобразуем цвета для выходного изображения
                color[1] = convertRGBToOutput(X_(0, pixel++));
                color[2] = convertRGBToOutput(X_(0, pixel++));
                if (i < imageWidth && j < imageHeight){	//пока не вышли за пределы изображения
                    image.draw_point(i,j,color);		//рисуем точку нужного цвета
                }
            }
        }
    }
    image.save("result_images/output_image2.bmp");	//сохраняем выходное изображение
}
//восстановление цветов для формирования выходного изображения
int ImageModel::convertRGBToOutput(double color){
    double ans = (255 * (color + 1) / 2);
    if (ans < 0){
        ans = 0;
    }
    if (ans > 255){
        ans = 255;
    }
    return (int)ans;
}
//суммарная среднеквадратическая ошибка для прямоугольника
double ImageModel::getErrorDegree(mat deltaX){
    double e=0;
    for (int i = 0; i < nmRGB; i++) {
        e += pow(deltaX(0, i), 2);		//суммарная среднеквадратическая ошибка для прямоугольника
    }
    return e;
}
//заполнение матриц весов псевдослучайными числами
void ImageModel::createWeightMatrix(){
    srand (time(NULL));
    W = randu<mat>(nmRGB,p);
    for (int i = 0; i < nmRGB; i++){
        for (int j = 0; j < p; j++)
            W(i,j) = (((double)rand() / RAND_MAX)*2 - 1 )*0.1;;
    }
    W_ = W.t(); //транспонированная матрица W
}
//получаем значение цвета пикселя для дальнейших преобразований
double ImageModel::convertColor(int color){
    return ((2.0 * color / 255) - 1);
}



