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
void detectCarPedestrian(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain);
Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold, double sigma);

int main(int argc, char *argv[]) {

  vector<Mat> data;

  Mat image;

  bool isEmpty, isOnA, isOnB, isOnC, isBarrier, isTrain;

  double number = 0;

  size_t numberOfImage;

  //Loading all the images to be tested
  for (int i = 1; i < argc; i++) {
    image = imread(argv[i]);
    if (image.empty()) continue;
    data.push_back(image);
  }

  numberOfImage = argc - 1;

  for (size_t imNumber = 0; imNumber < numberOfImage; imNumber++) {

    //Setting everything for the image;
    isEmpty = false;
    isOnA = false;
    isOnB = false;
    isOnC = false;
    isBarrier = false;
    isTrain = false;

    std::cout << "New Image :" << std::endl;

    //Train Detection
    isTrain = detectTrain(data[imNumber]);

    //Car detection
    detectCarPedestrian(data[imNumber], &isOnA, &isOnB, &isOnC, &isTrain);

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
      number++;
    }
    if (isBarrier) {
      cout << argv[imNumber + 1] << " : event 4" << endl;
    }
    if (isTrain) {
      cout << argv[imNumber + 1] << " : event 5" << endl;
    }
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
  bitwise_and(imageGray, maskZoneA, imageGray);
  reshapedImage = imageGray.reshape(1,1);
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

  Mat imageGray, contoursB;
  Mat maskBarrier = imread("Images/Masks/maskBarrier.png",CV_LOAD_IMAGE_GRAYSCALE);

  std::vector<Vec4i> lines(200); //If not on windows, (200) can be removed, if not big enough can cause problems

  Point startBarrier = Point(23, 265);

  cvtColor(image, imageGray, CV_BGR2GRAY);
  bitwise_and(imageGray,maskBarrier,imageGray);
  Canny(imageGray,contoursB,150,400);

  lines.clear();
  HoughLinesP(contoursB, lines, 1, PI/180, 10, 100, 20);

  for (int i = 0; i < static_cast<int>(lines.size()); i++) {
    line(imageGray, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0), 8);
    for (int pixelNumber = 190; pixelNumber < 285; pixelNumber++) {
      startBarrier.y = pixelNumber;
      if (norm(startBarrier - Point(lines[i][0], lines[i][1])) < 32 && (lines[i][2] - lines[i][0]) > 25) {
        return true;
      }
    }
  }
  return false;
}

void detectCarPedestrian(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain) {

  Mat maskLetters = imread("Images/Masks/maskLetters.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat emptySource = imread("Images/Test/Empty/lc-00010.png",CV_LOAD_IMAGE_COLOR);
  Mat rail = imread("Images/Masks/rail.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB = imread("Images/Masks/maskZoneB.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC = imread("Images/Masks/maskZoneC.png",CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB1 = imread("Images/Masks/maskZoneB1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC1 = imread("Images/Masks/maskZoneC1.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB2 = imread("Images/Masks/maskZoneB2.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC2 = imread("Images/Masks/maskZoneC2.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskTrain = imread("Images/Masks/maskTrain.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskPedestrian = imread("Images/Masks/maskPedestrian.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat result(image.size(), CV_8U, Scalar(255));

  Mat carImage, thresholdedImage, carGray, emptySourceGray, maskLettersInv, carGrayWhite, cannyOutput, cannyOutputWhite, carGrayTresholded, carGrayWhiteTresholded;

  static vector<vector<Point>> contoursCar, contoursCarWhite;
  vector<vector<Point>> contoursCarSelected,  contoursCarFinal, contourPedestrian, contourPedestrianFinal;
  vector<Point> poly(200), polyPedestrian(200);

  int thresh = 75, cmin= 200, cmax = 2000, cminP = 150, cmaxP = 800;

  vector<vector<Point>>::iterator cIt;
  vector<Point> momentsCars;

  bool currentB, currentC;
  float radius;
  Point2f center;
  Point point;

  int morph_elem = 0, morph_size = 5;
  Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));

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
  carGrayWhiteTresholded = getThresholdedImage(carGrayWhite, 200, 255, 1);
  bitwise_and(carGrayWhiteTresholded,maskLetters,carGrayWhiteTresholded);
  carGrayWhiteTresholded = carGrayWhiteTresholded - rail;
  bitwise_not(carGrayWhiteTresholded,carGrayWhiteTresholded);
  carGrayWhiteTresholded += maskLettersInv;

  bitwise_and(carGrayWhiteTresholded, carGrayTresholded, thresholdedImage);
 
  morphologyEx(thresholdedImage, thresholdedImage, MORPH_OPEN, element);

  //Apply the canny algorithm
  Canny(thresholdedImage, cannyOutput, thresh, thresh*2, 5);
  findContours(cannyOutput, contoursCar, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  for (cIt = contoursCar.begin(); cIt < contoursCar.end(); cIt++) {
    if (static_cast<int>(cIt->size()) > cmin) {
      contoursCarSelected.push_back(*cIt);
    }
    if (static_cast<int>(cIt->size()) > cminP && static_cast<int>(cIt->size()) < cmaxP) {
      contourPedestrian.push_back(*cIt);
    }
  }

  for (size_t i = 0; i < contoursCarSelected.size(); i++) {
    currentB = false;
    currentC = false;
    convexHull(Mat(contoursCarSelected[i]),poly);
    contoursCarFinal.push_back(poly);
    Moments mom= moments(Mat(poly));
    if (mom.m00 > 0) {
      minEnclosingCircle(Mat(poly),center,radius);
      if (((PI *radius *radius) / contourArea(poly)) < 6 && contourArea(poly) > 800) {
        point = Point(static_cast<int>(mom.m10/mom.m00), static_cast<int>(mom.m01/mom.m00));
        if (!*isTrain) {
          if (maskZoneB.at<uchar>(point.y, point.x) == 255 && !currentB) {
            *isOnB = true;
            currentB = true;
          }

          if (maskZoneA.at<uchar>(point.y,point.x) == 255 && !*isOnA) {
            *isOnA = true;
          }

          if (maskZoneC.at<uchar>(point.y,point.x) == 255 && !currentC) {
            *isOnC = true;
            currentC = true;
          }

          if (!*isOnA && !*isTrain && (currentB || currentC)) {
            for (size_t j = 0; j < poly.size(); j++) {
              if (maskZoneA.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnA = true;
              }
            }
          }
          if (!currentB && !currentC) {
            for (size_t j = 0; j < poly.size(); j++) {
              if (maskZoneB1.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnB = true;
                currentB = true;
              }
              if (maskZoneC2.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnC = true;
                currentC = true;
              }
              if (maskZoneC1.at<uchar>(poly[j].y, poly[j].x) == 255 && !currentB) {
                *isOnC = true;
                currentC = true;
              }
              if (maskZoneB2.at<uchar>(poly[j].y, poly[j].x) == 255 && !currentC) {
                *isOnB = true;
                currentB = true;
              }
            }
          }
        } else {
          if (maskTrain.at<uchar>(point.y, point.x) == 255 && !*isOnC) {
            *isOnB = true;
          }
        }
      } else {
        contoursCarFinal.pop_back();
      }
    } else {
      contoursCarFinal.pop_back();
    }
  }

  for (size_t i = 0; i < contourPedestrian.size(); i++) {
    convexHull(Mat(contourPedestrian[i]), polyPedestrian);
    contourPedestrianFinal.push_back(polyPedestrian);
    Moments mom = moments(Mat(polyPedestrian));
    if (mom.m00 > 0) {
      minEnclosingCircle(Mat(polyPedestrian), center, radius);
      if (((PI *radius *radius) / contourArea(polyPedestrian)) < 4 && contourArea(polyPedestrian) > 400) {
        point = Point(static_cast<int>(mom.m10 / mom.m00), static_cast<int>(mom.m01 / mom.m00));
        if (!*isTrain) {
          if (maskPedestrian.at<uchar>(point.y, point.x) == 255 && !*isOnA) {
            *isOnA = true;
          }
        }
      }
      else {
        contourPedestrianFinal.pop_back();
      }
    } else {
      contourPedestrianFinal.pop_back();
    }
  }
  //drawContours(thresholdedImage, contoursCarFinal, -1, Scalar(150),5);
  //imshow("thresholdedImage", thresholdedImage);
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
