/*
 * This program allow the user to analyse one or multiple images and detect event occuring.
 * To use this program : 
 *    - programName pathToImage (bin\ImageAnalysis.exe Images\Test\OnTrack\lc-00006.png)
 *    - programName pathToDirectory (bin\ImageAnalysis.exe Images\Test\OnTrack\*png)
 *
 * WARNING :
 * Be aware that the program needs to access images/files under the following pathes : Images/path/to/image Models/path/to/model.
 * Thus the corresponding directories need to be placed at the correct location.
 * Also this program require to create a svm model with the main function within the makeModel.cpp file.
 *
 * This program was made using the lectures notes from Image Analysis, the provided files and the opencv documentation and examples.
 *
 * \author Benjamin Deguerre
*/

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

  //Boolean for the different events.
  bool isEmpty, isOnA, isOnB, isOnC, isBarrier, isTrain;

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

    //Car & pedestrian detection
    detectCarPedestrian(data[imNumber], &isOnA, &isOnB, &isOnC, &isTrain);

    //Barrier detection
    isBarrier = detectBarrier(data[imNumber], isTrain);
 
    //Output the results to the console prompt.
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
    }
  }
  return 0;
}

bool detectTrain(Mat image) {

  Mat imageGray, maskedImageGray, reshapedImage;
  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE);

  CvSVM SVMTrain;
  float svmResult;

  //Preparing the image for the SVM classifier
  cvtColor(image, imageGray, CV_BGR2GRAY);
  bitwise_and(imageGray, maskZoneA, imageGray);
  reshapedImage = imageGray.reshape(1,1);
  reshapedImage.convertTo(reshapedImage, CV_32F);

  //Loading the model.
  SVMTrain.load("Models/trainModel.xml");

  //Predicting the result.
  svmResult = SVMTrain.predict(reshapedImage, false);
  if (svmResult == 1) {
    return true;
  }
  return false;
}

bool detectBarrier(Mat image, bool isTrain) {

  //If there is a train then the barrier must be down.
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
  //Detecting the lines
  HoughLinesP(contoursB, lines, 1, PI/180, 10, 100, 20);

  //We iterate through the lines and look for some with a start point close to the left edge of the image.
  for (int i = 0; i < static_cast<int>(lines.size()); i++) {
    line(imageGray, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0), 8);
    for (int pixelNumber = 190; pixelNumber < 285; pixelNumber++) {
      startBarrier.y = pixelNumber;
      //if line close to the edge and not straigh up
      if (norm(startBarrier - Point(lines[i][0], lines[i][1])) < 32 && (lines[i][2] - lines[i][0]) > 25) {
        return true;
      }
    }
  }
  return false;
}

void detectCarPedestrian(Mat image, bool *isOnA, bool *isOnB, bool *isOnC, bool *isTrain) {

  //Loading all of the masks.
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

  Mat carImage, thresholdedImage, carGray, emptySourceGray, maskLettersInv, carGrayWhite, cannyOutput, cannyOutputWhite, carGrayThresholded, carGrayWhiteThresholded;

  static vector<vector<Point>> contoursCar, contoursCarWhite;
  vector<vector<Point>> contoursCarSelected,  contoursCarFinal, contourPedestrian, contourPedestrianFinal;
  vector<Point> poly(200), polyPedestrian(200);

  //Diffrent threshold needed
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
  //First preparing the image
  image.copyTo(carImage);
  carImage = emptySource - carImage;
  cvtColor(carImage, carGray, CV_BGR2GRAY);
  carGray = carGray - rail;

  //Then thresold it to keep the cars
  carGrayThresholded = getThresholdedImage(carGray, 200, 255, 1);

  //Applying different masks and inversing the colors
  bitwise_and(carGrayThresholded,maskLetters,carGrayThresholded);
  bitwise_not(carGrayThresholded,carGrayThresholded);
  carGrayThresholded += maskLettersInv;

  //Then the same for the white ones
  cvtColor(image, carGrayWhite, CV_BGR2GRAY);
  carGrayWhite = carGrayWhite - emptySourceGray;

  //logarithm transform && threshold
  carGrayWhiteThresholded = getThresholdedImage(carGrayWhite, 200, 255, 1);

  //Applying different masks and inversing the colors
  bitwise_and(carGrayWhiteThresholded,maskLetters,carGrayWhiteThresholded);
  carGrayWhiteThresholded = carGrayWhiteThresholded - rail;
  bitwise_not(carGrayWhiteThresholded,carGrayWhiteThresholded);
  carGrayWhiteThresholded += maskLettersInv;

  //Combining the two images (white and non-white cars).
  bitwise_and(carGrayWhiteThresholded, carGrayThresholded, thresholdedImage);
 
  //Opening to make the cars easier to detect.
  morphologyEx(thresholdedImage, thresholdedImage, MORPH_OPEN, element);

  //Apply the canny algorithm
  Canny(thresholdedImage, cannyOutput, thresh, thresh*2, 5);
  findContours(cannyOutput, contoursCar, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  //Selecting the contours to analyse.
  for (cIt = contoursCar.begin(); cIt < contoursCar.end(); cIt++) {
    if (static_cast<int>(cIt->size()) > cmin) {
      contoursCarSelected.push_back(*cIt);
    }
    if (static_cast<int>(cIt->size()) > cminP && static_cast<int>(cIt->size()) < cmaxP) {
      contourPedestrian.push_back(*cIt);
    }
  }

  //Iterating through the selected contours to trigger the events (cars)
  for (size_t i = 0; i < contoursCarSelected.size(); i++) {
    //Setting the detected event for the current contour to false.
    currentB = false;
    currentC = false;
    
    //Approximating the shape of the contour with a convexHull algorithm.
    convexHull(Mat(contoursCarSelected[i]),poly);

    //Calculating the moments of the shape.
    Moments mom= moments(Mat(poly));

    //To avoid zero division.
    if (mom.m00 > 0) {
      //Getting the minimum enclosing circle for the shape.
      minEnclosingCircle(Mat(poly), center, radius);
      //If the ratio between the area of minEnclosingCircle and the area of the poly to high then not a car (long shape),
      //if area of the poly too small, also noise.
      if (((PI *radius *radius) / contourArea(poly)) < 6 && contourArea(poly) > 800) {
        //Center of the shape.
        point = Point(static_cast<int>(mom.m10 / mom.m00), static_cast<int>(mom.m01 / mom.m00));
        //Using different masks depending on the presence of a train.
        if (!*isTrain) {
          //If center in the masks
          if (!currentB && maskZoneB.at<uchar>(point.y, point.x) == 255) {
            *isOnB = true;
            currentB = true;
          }

          if (!*isOnA && maskZoneA.at<uchar>(point.y, point.x) == 255) {
            *isOnA = true;
          }

          if (!currentC && maskZoneC.at<uchar>(point.y, point.x) == 255) {
            *isOnC = true;
            currentC = true;
          }

          //If center not in the masks, the contours might be.
          //If the shape is not either on B or C, contours cannot be on A.
          if (!*isOnA && !*isTrain && (currentB || currentC)) {
            for (size_t j = 0; j < poly.size(); j++) {
              if (maskZoneA.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnA = true;
              }
            }
          }
          //If center on the shape not on B and C, the contours could be.
          if (!currentB && !currentC) {
            for (size_t j = 0; j < poly.size(); j++) {
              //First chacking for the zones at the bottom to reduce false positives.
              if (maskZoneB1.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnB = true;
                currentB = true;
              }
              if (maskZoneC2.at<uchar>(poly[j].y, poly[j].x) == 255) {
                *isOnC = true;
                currentC = true;
              }
              //Then the zones at the top.
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
        }
        else {
          if (maskTrain.at<uchar>(point.y, point.x) == 255 && !*isOnC) {
            *isOnB = true;
          }
        }
      }
    }
  }

  //Checking for pedestrians.
  for (size_t i = 0; i < contourPedestrian.size(); i++) {
    convexHull(Mat(contourPedestrian[i]), polyPedestrian);
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
    }
  }
}

Mat getThresholdedImage(Mat image, int lowThreshold, int highThreshold, double sigma) {
  double max, min;
  int cLog;

  //Apply logarithmic transform and threshold the image.
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
