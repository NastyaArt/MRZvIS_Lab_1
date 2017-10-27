#include <QCoreApplication>

#include <iostream>
#include <CImg.h>
#include <vector>
#include <armadillo>

#define ARMA_DONT_USE_WRAPPER

#include "rectanglemodel.h"
#include "imagemodel.h"

using namespace std;
using namespace cimg_library;
using namespace arma;

int main()
{
    /*CImg<unsigned char> src("user2.bmp");
    mat W;
    */
    ImageModel model("user2.bmp");
    model.run();
    model.createOutputImage();
    return 0;
}
