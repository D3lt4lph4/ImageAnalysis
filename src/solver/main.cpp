#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>

# define PI 3.14159265358979323846

using namespace cv;
using namespace std;

bool detectTrain(Mat image);
bool detectBarrier(Mat image, bool isTrain);
void detectCar(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain);
Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold, double sigma);
bool detectPedestrian(Mat image);

int main(int argc, char *argv[]) {

  string file = argv[1];

  static vector<String> fileName;

  vector<Mat> data;

  //Loading the masks to be used
  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB = imread("Images/Masks/maskZoneB.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC = imread("Images/Masks/maskZoneC.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskLetters = imread("Images/Masks/maskLetters.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat emptySource = imread("Images/Test/Empty/lc-00010.png",CV_LOAD_IMAGE_COLOR);


  bool isEmpty = false, isOnA = false, isOnB = false, isOnC = false, isBarrier = false, isTrain = false, noCars = true;


  size_t numberOfImage;

  vector<float> labels;
  vector<Mat> vectorData;

  Mat im, gray, gray2,image_f, maskedImage, image,temp;

  Mat test = imread("Images/Test/Entering/lc-00352.png",CV_LOAD_IMAGE_COLOR);

  float number =0;

  //Loading all the images to be tested
  for (int i = 1; i < argc; i++) {
    image = imread(argv[i]);
    if (image.empty()) continue;
    data.push_back(image);
  }

  numberOfImage = argc - 1;

  for (size_t imNumber = 0; imNumber < numberOfImage; imNumber++) {
    std::cout << "New Image :" << std::endl;

    //Train Detection
    isTrain = detectTrain(data[imNumber]);
    
    //Car detection
    detectCar(data[imNumber], &isOnA, &isOnB, &isOnC, &isTrain);

    //Pedestrian detection


    //Barrier detection
    isBarrier = detectBarrier(data[imNumber], isTrain);

    //If nothing detected then must be empty
    if (!isOnA && !isOnB && !isOnC && !isBarrier && !isTrain) {
      isEmpty = true;
      cout << argv[imNumber+1] << " : event 0" << endl;
    }
    if (isOnA) {
      cout << argv[imNumber + 1] << " : event 1" << endl;
    }
    if (isOnB) {
      cout << argv[imNumber + 1] << " : event 2" << endl;
    }
    if (isOnC) {
      cout << argv[imNumber + 1] << " : event 3" << endl;
    }
    if (isBarrier) {
      cout << argv[imNumber + 1] << " : event 4" << endl;
    }
    if (isTrain) {
      cout << argv[imNumber + 1] << " : event 5" << endl;
      number++;
    }

    isEmpty = false;
    isOnA = false;
    isOnB = false;
    isOnC = false;
    isBarrier = false;
    isTrain = false;
    noCars = true;
 /*   imshow("contour", data[imNumber]);
    waitKey(0);*/
  }
  std::cout << "well found percentage :" << number / numberOfImage * 100 << std::endl;
  std::cout << "detected :" << number << std::endl;
  std::cout << "number of image : " << numberOfImage << std::endl;
  return 0;
}

bool detectTrain(Mat image) {

  Mat imageGray, maskedImageGray, reshapedImage;
  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE);

  CvSVM SVMTrain;
  float svmResult;

  cvtColor(image, imageGray, CV_BGR2GRAY);
  imageGray.copyTo(maskedImageGray, maskZoneA);
  reshapedImage = maskedImageGray.reshape(1,1);
  reshapedImage.convertTo(reshapedImage, CV_32F);
  SVMTrain.load("Models/trainModel.xml");
  svmResult = SVMTrain.predict(reshapedImage, false);
  if (svmResult == 1) {
    return true;
  }
  return false;
}

bool detectBarrier(Mat image, bool isTrain) {
  
  if (isTrain) {
    return true;
  }
  Mat imageGray;
  Mat contoursB;
  Mat maskBarrier = imread("Images/Masks/maskBarrier.png",CV_LOAD_IMAGE_GRAYSCALE);

  std::vector<Vec4i> lines(200);

  Point startBarrier = Point(23, 265);
  int pixelNumber;

  cvtColor(image, imageGray, CV_BGR2GRAY);
  bitwise_and(imageGray,maskBarrier,imageGray);
  Canny(imageGray,contoursB,125,350);

  lines.clear();
  HoughLinesP(contoursB, lines, 1, PI/180, 10, 100, 20);
  circle(imageGray, startBarrier, 2, Scalar(0),2);


  for (int i = 0; i < static_cast<int>(lines.size()); i++) {
    pixelNumber = 190;
    do {
      startBarrier.y = pixelNumber;
      if (norm(startBarrier - Point(lines[i][0], lines[i][1])) < 20 && (lines[i][2] - lines[i][0]) > 25) {
        return true;
      }
      pixelNumber++;
    } while (pixelNumber < 285);
  }
  return false;
}

void detectCar(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain) {

  Mat maskLetters = imread("Images/Masks/maskLetters.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat emptySource = imread("Images/Test/Empty/lc-00010.png",CV_LOAD_IMAGE_COLOR);
  Mat rail = imread("Images/Masks/rail.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB = imread("Images/Masks/maskZoneB.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC = imread("Images/Masks/maskZoneC.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskTrain = imread("Images/Masks/maskTrain.png", CV_LOAD_IMAGE_GRAYSCALE);

  Mat carImage, carGray, emptySourceGray, maskLettersInv, carGrayWhite, cannyOutput, cannyOutputWhite, carGrayTresholded, carGrayWhiteTresholded;

  static vector<vector<Point>> contoursCar, contoursCarSelected, contoursCarWhite, contoursCarFinal;

  vector<Point> poly(200);

  int thresh = 75, cmin= 200, cmax = 2000;

  vector<vector<Point>>::iterator cIt;
  vector<Point> momentsCars;

  float radius;
  Point2f center;
  Point point;

  cvtColor(emptySource, emptySourceGray, CV_BGR2GRAY);
  //Getting the inverse of the mask to remove the black part at the top
  bitwise_not(maskLetters, maskLettersInv);

  //We handle the image for detecting non white cars
  image.copyTo(carImage);
  carImage = emptySource - carImage;
  cvtColor(carImage, carGray, CV_BGR2GRAY);

  carGray = carGray - rail;
  carGrayTresholded = getThresholdedImage(carGray, 200, 255, 1);

  bitwise_and(carGrayTresholded,maskLetters,carGrayTresholded);
  bitwise_not(carGrayTresholded,carGrayTresholded);
  carGrayTresholded += maskLettersInv;

  //Then for the white ones
  cvtColor(image, carGrayWhite, CV_BGR2GRAY);
  carGrayWhite = carGrayWhite - emptySourceGray;

  //logarithm transform && threshold
  carGrayWhiteTresholded = getThresholdedImage(carGrayWhite, 190, 255, 1);
  bitwise_and(carGrayWhiteTresholded,maskLetters,carGrayWhiteTresholded);
  carGrayWhiteTresholded = carGrayWhiteTresholded - rail;
  bitwise_not(carGrayWhiteTresholded,carGrayWhiteTresholded);
  carGrayWhiteTresholded += maskLettersInv;
  //Apply the canny algorithm
  Canny(carGrayTresholded, cannyOutput, thresh, thresh*2, 5);
  Canny(carGrayWhiteTresholded, cannyOutputWhite, thresh, thresh*2, 5);
  findContours(cannyOutput, contoursCar, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  findContours(cannyOutputWhite, contoursCarWhite, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  Mat result(image.size(),CV_8U,Scalar(255));
  for (cIt = contoursCar.begin(); cIt < contoursCar.end(); cIt++) {
    if (static_cast<int>(cIt->size()) > cmin && static_cast<int>(cIt->size()) < cmax) {
      contoursCarSelected.push_back(*cIt);
    }
  }
  for (cIt = contoursCarWhite.begin(); cIt < contoursCarWhite.end(); cIt++) {
    if (static_cast<int>(cIt->size()) > cmin && static_cast<int>(cIt->size()) < cmax) {
      contoursCarSelected.push_back(*cIt);
    }
  }
  for (size_t i = 0; i < contoursCarSelected.size(); i++) {
    convexHull(Mat(contoursCarSelected[i]),poly);
    contoursCarFinal.push_back(poly);
    Moments mom= moments(Mat(poly));
    if (mom.m00 > 0) {
      minEnclosingCircle(Mat(poly),center,radius);
      if (((PI *radius *radius) / contourArea(poly)) < 8 && contourArea(poly) > 800) {

        point = Point(static_cast<int>(mom.m10/mom.m00), static_cast<int>(mom.m01/mom.m00));
        

        if (!*isTrain) {
          if (maskZoneB.at<uchar>(point.y, point.x) == 255 && !*isOnB) {
            *isOnB = true;
          }

          if (maskZoneA.at<uchar>(point.y,point.x) == 255 && !*isOnA) {
            *isOnA = true;
          }

          if (maskZoneC.at<uchar>(point.y,point.x) == 255 && !*isOnC) {
            *isOnC = true;
          }
        }
        else {
          if (maskTrain.at<uchar>(point.y, point.x) == 255 && !*isOnC) {
            *isOnB = true;
          }
        }
      } else {
        contoursCarFinal.pop_back();
      }
      circle(result, point, 2, Scalar(0),2); // to be removed
      drawContours(result, contoursCarFinal, -1, Scalar(0), 2); // to be removed
    } else {
      contoursCarFinal.pop_back();
    }
    if (!*isOnA && !*isTrain) {
      for (size_t j = 0; j < contoursCarSelected[i].size(); j++) {
        if (maskZoneA.at<uchar>(contoursCarSelected[i][j].y, contoursCarSelected[i][j].x) == 255 && !*isOnA) {
          *isOnA = true;
        }
      }
    }

  }
  //drawContours(result, contoursCarSelected, -1, Scalar(0), 2);
  //imshow("contour", result);
  //waitKey(0);
}

Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold, double sigma) {
  double max, min;
  int cLog;

  minMaxLoc(image, &min,&max);
  cLog = static_cast<int>(255 / log(sigma + max));
  image = 1+image;
  image.convertTo(image,CV_32F);
  log(image,image);
  image = cLog*image;
  normalize(image,image,0,255,NORM_MINMAX);
  convertScaleAbs(image,image);
  threshold(image, image, lowThreshold, highThreshold,THRESH_BINARY);
  return image;
}

bool detectPedestrian(Mat image) {
  return true;
}
