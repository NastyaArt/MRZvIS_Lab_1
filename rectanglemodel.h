#ifndef RECTANGLEMODEL_H
#define RECTANGLEMODEL_H

#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;

class RectangleModel
{
private:
    int startX;  //начальные координаты, т.е. с какого пикселя начинается прямоугольник - W
    int startY;	 // - H
    vector<double> vectorX;  //вектор значения цветов
    mat X;	//матрица цветов
public:
    RectangleModel(int startX, int startY);  //задаем начальные координаты прямоугольника в конструкторе
    void createMatrixX();					//создать матрицу по вектору
    void addElement(double newElement);		//добавить элемент в вектор цветов
    int getStartX();						//получить начальные координаты по X
    void setStartX(int start);				//не юзается походу
    int getStartY();						//получить начальные координаты по Y
    void setStartY(int start);				//не юзается походу
    vector<double> getVectorX();			//получить вектор цветов
    void setVectorX(vector<double> vector);		//не юзается походу
    mat getX();								//получить матрицу цветов
    void setX(mat x);						//не юзается походу
};

#endif // RECTANGLEMODEL_H
