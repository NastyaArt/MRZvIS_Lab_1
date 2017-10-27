#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H

#include <iostream>
#include <CImg.h>
#include <vector>
#include <armadillo>

#include "rectanglemodel.h"

using namespace std;
using namespace cimg_library;
using namespace arma;

class ImageModel
{
private:
    int RGB = 3;	//R+G+B=3
    int imageWidth;  //ширина изображения в пикселях
    int imageHeight;  //высота изображения в пикселях
    int n; //ширина прямоугольника
    int m; //высота прямоугольника
    int p; //число нейронов второго слоя
    double e; //максимальная допустимая ошибка 
    int L;  //количество прямоугольников
    int nmRGB;  //nmRGB = n * m * RGB
    double a;  //коэффициент обучения
    vector<RectangleModel> rectangleModelList;  //список прямоугольников
    mat W;  //веса на первом слое
    mat W_; //веса на втором слое
    int convertRGBToOutput(double rgb);  //восстановление цветов для формирования выходного изображения
    double getErrorDegree(mat deltaX); //суммарная среднеквадратическая ошибка для прямоугольника
    void createWeightMatrix();  //заполнение матриц весов псевдослучайными числами
    double convertColor(int color); //получаем значение цвета пикселя для дальнейших преобразований
    void normalizeMatrix(mat matrix);  //нормализация матрицы
    void normalizeMatrixs(); //нормализация весов на первом и втором слое
    double adaptiveLearningStep(mat matrix); //расчет адаптивного шага обучения
public:
    ImageModel(char const * patch);
    void run();		//запуск работы с изображением
    void initANN();  //ввод входных параметров
    void createOutputImage();  //формирование выходного изображения
};

#endif // IMAGEMODEL_H
