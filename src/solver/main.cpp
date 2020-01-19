#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.14159265358979323846

using namespace cv;
using namespace cv::ml;

using namespace std;

bool detectTrain(Mat image);

bool detectBarrier(Mat image, bool isTrain);

void detectCar(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain);

Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold,
                        double sigma);

bool detectPedestrian(Mat image);

int main(int argc, char *argv[]) {
  vector<String> fileName;

  vector<Mat> data;

  // Loading the masks to be used
  Mat maskZoneA = imread("images/masks/maskZoneA.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB = imread("images/masks/maskZoneB.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC = imread("images/masks/maskZoneC.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskLetters =
      imread("images/masks/maskLetters.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat emptySource =
      imread("images/test/empty/lc-00010.png", CV_LOAD_IMAGE_COLOR);

  bool isEmpty = false, isOnA = false, isOnB = false, isOnC = false,
       isBarrier = false, isTrain = false, noCars = true;

  size_t numberOfImage;

  vector<float> labels;
  vector<Mat> vectorData;

  Mat im, gray, gray2, image_f, maskedImage, image, temp;

  int rows, cols;
  float number_zero = 0, number_one = 0, number_two = 0, number_three = 0,
        number_four = 0, number_five = 0;

  // Loading all the images to be tested
  glob(argv[1], fileName, false);

  for (size_t k = 0; k < fileName.size(); ++k) {
    image = imread(fileName[k]);
    if (image.empty()) continue;
    data.push_back(image);
  }

  numberOfImage = data.size();

  for (size_t imNumber = 0; imNumber < numberOfImage; imNumber++) {
    // std::cout << "New Image :" << std::endl;

    // Train Detection
    isTrain = detectTrain(data[imNumber]);

    // Car detection
    detectCar(data[imNumber], &isOnA, &isOnB, &isOnC, &isTrain);

    // Pedestrian detection

    // Barrier detection
    isBarrier = detectBarrier(data[imNumber], isTrain);

    // If nothing detected then must be empty
    if (!isOnA && !isOnB && !isOnC && !isBarrier && !isTrain) {
      isEmpty = true;
      number_zero++;
      // cout << fileName[imNumber] << " : event 0" << endl;
    }
    if (isOnA) {
      // cout << fileName[imNumber] << " : event 1" << endl;
      number_one++;
      // imshow("image", data[imNumber]);
      // waitKey(0);
    }
    if (isOnB) {
      number_two++;
      // cout << fileName[imNumber] << " : event 2" << endl;
    }
    if (isOnC) {
      number_three++;
      // cout << fileName[imNumber] << " : event 3" << endl;
    }
    if (isBarrier) {
      number_four++;
      // cout << fileName[imNumber] << " : event 4" << endl;
    }
    if (isTrain) {
      number_five++;
      // cout << fileName[imNumber] << " : event 5" << endl;
    }

    isEmpty = false;
    isOnA = false;
    isOnB = false;
    isOnC = false;
    isBarrier = false;
    isTrain = false;
    noCars = true;
  }
  std::cout << "Number of images processed: " << numberOfImage << std::endl;
  std::cout << "Number of event 0 found: " << number_zero << std::endl;
  std::cout << "Number of event 1 found: " << number_one << std::endl;
  std::cout << "Number of event 2 found: " << number_two << std::endl;
  std::cout << "Number of event 3 found: " << number_three << std::endl;
  std::cout << "Number of event 4 found: " << number_four << std::endl;
  std::cout << "Number of event 5 found: " << number_five << std::endl;
  return 0;
}

bool detectTrain(Mat image) {
  Mat imageGray, maskedImageGray, reshapedImage;
  Mat maskZoneA = imread("images/masks/maskZoneA.png", CV_LOAD_IMAGE_GRAYSCALE);

  Ptr<SVM> svm = Algorithm::load<ml::SVM>("models/trainModel.xml");
  float svmResult;

  cvtColor(image, imageGray, CV_BGR2GRAY);
  imageGray.copyTo(maskedImageGray, maskZoneA);
  reshapedImage = maskedImageGray.reshape(1, 1);
  reshapedImage.convertTo(reshapedImage, CV_32F);

  svmResult = svm->predict(reshapedImage);

  if (svmResult == 1) {
    return true;
  }
  return false;
}

bool detectBarrier(Mat image, bool isTrain) {
  // Apply Canny algorithm
  Mat imageGray, contoursB;
  Mat maskBarrier =
      imread("images/masks/maskBarrier.png", CV_LOAD_IMAGE_GRAYSCALE);

  std::vector<Vec4i> lines;

  Point startBarrier = Point(23, 265);
  int pixelNumber;

  cvtColor(image, imageGray, CV_BGR2GRAY);
  bitwise_and(imageGray, maskBarrier, imageGray);
  Canny(imageGray, contoursB, 125, 350);

  lines.clear();
  HoughLinesP(contoursB, lines, 1, PI / 180, 10, 100, 20);
  circle(imageGray, startBarrier, 2, Scalar(0), 2);

  if (!isTrain) {
    for (int i = 0; i < lines.size(); i++) {
      pixelNumber = 190;
      do {
        startBarrier.y = pixelNumber;
        if (norm(startBarrier - Point(lines[i][0], lines[i][1])) < 20 &&
            (lines[i][2] - lines[i][0]) > 25) {
          return true;
        }
        pixelNumber++;
      } while (pixelNumber < 285);
    }
  } else {
    for (int i = 0; i < lines.size(); i++) {
      startBarrier.y = 278;
      if (norm(startBarrier - Point(lines[i][0], lines[i][1])) < 4 &&
          (lines[i][2] - lines[i][0]) > 25) {
        return true;
      }
    }
  }
  return false;
}

void detectCar(Mat image, bool *isOnA, bool *isOnB, bool *isOnC,
               bool *isTrain) {
  Mat maskLetters =
      imread("images/masks/maskLetters.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat emptySource =
      imread("images/test/empty/lc-00010.png", CV_LOAD_IMAGE_COLOR);
  Mat rail = imread("images/masks/rail.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneA = imread("images/masks/maskZoneA.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneB = imread("images/masks/maskZoneB.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat maskZoneC = imread("images/masks/maskZoneC.png", CV_LOAD_IMAGE_GRAYSCALE);

  Mat carImage, carGray, emptySourceGray, maskLettersInv, carGrayWhite,
      cannyOutput, cannyOutputWhite, carGrayTresholded, carGrayWhiteTresholded;

  vector<vector<Point>> contoursCar, contoursCarWhite, contoursCarFinal;
  vector<Point> poly;

  int thresh = 75, cmin = 200, cmax = 2000;

  vector<vector<Point>>::iterator cIt;
  vector<Point> momentsCars;

  Point point;

  cvtColor(emptySource, emptySourceGray, CV_BGR2GRAY);
  // Getting the inverse of the mask to remove the black part at the top
  bitwise_not(maskLetters, maskLettersInv);

  // We handle the image for detecting non white cars
  image.copyTo(carImage);
  carImage = emptySource - carImage;
  cvtColor(carImage, carGray, CV_BGR2GRAY);

  carGray = carGray - rail;
  carGrayTresholded = getThresholdedImage(carGray, 200, 255, 1);

  bitwise_and(carGrayTresholded, maskLetters, carGrayTresholded);
  bitwise_not(carGrayTresholded, carGrayTresholded);
  carGrayTresholded += maskLettersInv;

  // Then for the white ones
  cvtColor(image, carGrayWhite, CV_BGR2GRAY);
  carGrayWhite = carGrayWhite - emptySourceGray;

  // logarithm transform && threshold
  carGrayWhiteTresholded = getThresholdedImage(carGrayWhite, 190, 255, 1);
  bitwise_and(carGrayWhiteTresholded, maskLetters, carGrayWhiteTresholded);
  carGrayWhiteTresholded = carGrayWhiteTresholded - rail;
  bitwise_not(carGrayWhiteTresholded, carGrayWhiteTresholded);
  carGrayWhiteTresholded += maskLettersInv;

  // Apply the canny algorithm
  Canny(carGrayTresholded, cannyOutput, thresh, thresh * 2, 5);
  Canny(carGrayWhiteTresholded, cannyOutputWhite, thresh, thresh * 2, 5);

  findContours(cannyOutput, contoursCar, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  findContours(cannyOutputWhite, contoursCarWhite, CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_NONE);

  Mat result(image.size(), CV_8U, Scalar(255));
  for (cIt = contoursCar.begin(); cIt < contoursCar.end(); cIt++) {
    if (cIt->size() < cmin || cIt->size() > cmax) {
      contoursCar.erase(cIt);
      cIt--;
    }
  }
  for (cIt = contoursCarWhite.begin(); cIt < contoursCarWhite.end(); cIt++) {
    if (cIt->size() > cmin && cIt->size() < cmax) {
      contoursCar.push_back(*cIt);
    }
  }

  for (size_t i = 0; i < contoursCar.size(); i++) {
    convexHull(Mat(contoursCar[i]), poly);
    contoursCarFinal.push_back(poly);
    Moments mom = moments(Mat(poly));
    if (mom.m00 > 0) {
      float radius;
      Point2f center;
      minEnclosingCircle(Mat(poly), center, radius);
      if (((PI * radius * radius) / contourArea(poly)) < 8 &&
          contourArea(poly) > 800) {
        point = Point(mom.m10 / mom.m00, mom.m01 / mom.m00);
        if (maskZoneB.at<uchar>(point.y, point.x) == 255 && !*isOnB) {
          *isOnB = true;
        }

        if (!*isTrain) {
          if (maskZoneA.at<uchar>(point.y, point.x) == 255 && !*isOnA) {
            *isOnA = true;
          }

          if (maskZoneC.at<uchar>(point.y, point.x) == 255 && !*isOnC) {
            *isOnC = true;
          }
        }
      } else {
        contoursCarFinal.pop_back();
      }
      circle(result, point, 2, Scalar(0), 2);  // to be removed
      drawContours(result, contoursCarFinal, -1, Scalar(0),
                   2);  // to be removed
    } else {
      contoursCarFinal.pop_back();
    }
    if (!*isOnA && !*isTrain) {
      for (size_t j = 0; j < contoursCar[i].size(); j++) {
        if (maskZoneA.at<uchar>(contoursCar[i][j].y, contoursCar[i][j].x) ==
                255 &&
            !*isOnA) {
          *isOnA = true;
        }
      }
    }
  }
}

Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold,
                        double sigma) {
  double max, min;
  int cLog;

  minMaxLoc(image, &min, &max);
  cLog = 255 / log(sigma + max);
  image = 1 + image;
  image.convertTo(image, CV_32F);
  log(image, image);
  image = cLog * image;
  normalize(image, image, 0, 255, NORM_MINMAX);
  convertScaleAbs(image, image);
  threshold(image, image, lowThreshold, highThreshold, THRESH_BINARY);
  return image;
}

bool detectPedestrian(Mat image) { return true; }
