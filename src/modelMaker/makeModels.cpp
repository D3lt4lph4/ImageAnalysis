#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void loadData(string path, vector<Mat> &data, vector<int> &label, int labelInt);

int main(int argc, char *argv[]) {

  cv::String trEm("Images/Training/Empty/*.png");
  cv::String trOT("Images/Training/OnTrack/*.png");
  cv::String trT("Images/Training/Train/*.png");

  cv::String tB("Images/Test/Barrier/*.png");
  cv::String tEm("Images/Test/Empty/*.png");
  cv::String tEn("Images/Test/Entering/*.png");
  cv::String tL("Images/Test/Leaving/*.png");
  cv::String tOT("Images/Test/OnTrack/*.png");
  cv::String tT("Images/Test/Train/*.png");

  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE), gray, maskedImage, reshapedImage;
  vector<int> labels;

  vector<Mat> data, vectorDataTrain, vectorDataEmpty;

  Ptr<ml::SVM> SVM = ml::SVM::create();
 

  Mat im , image,canny_output, original, yolo, yolo2,dst;
  Mat imTest, matTest;

  int rows, cols, thresh = 75;
  float PI = 3.14;

  loadData(trEm, data, labels, -1);
  loadData(trOT, data, labels, -1);
  loadData(trT, data, labels, 1);

  for (size_t k=0; k<data.size(); ++k)
  {
    cvtColor(data[k], gray, CV_BGR2GRAY);
    gray.copyTo(maskedImage, maskZoneA);
    reshapedImage = maskedImage.reshape(1,1);
    reshapedImage.convertTo(reshapedImage, CV_32F);
    vectorDataTrain.push_back(reshapedImage);
  }

  rows = labels.size();
  cols = reshapedImage.cols;

  // Set up training data
  Mat labelSetTrain(labels, false);
  Mat trainingDataSetTrain(rows, cols,CV_32F);

  for (size_t i = 0; i < rows; i++) {
    vectorDataTrain[i].copyTo(trainingDataSetTrain(Rect(0,i,cols,1)));
  }

  // Set up SVM's parameters
  SVM->setType(ml::SVM::C_SVC);
  SVM->setKernel(ml::SVM::LINEAR);

  Ptr<ml::TrainData> trainingData = cv::ml::TrainData::create(trainingDataSetTrain, ml::ROW_SAMPLE, labelSetTrain);
  // Train the SVM

  SVM->trainAuto(trainingData);
  SVM->save("Models/trainModel3.2.xml");
}

void loadData(string path, vector<Mat> &data, vector<int> &labels, int labelInt) {

  vector<cv::String> fn;
  Mat image;

  cv::glob(path,fn,false);

  for (size_t k=0; k<fn.size(); ++k)
  {
    image = cv::imread(fn[k]);
    if (image.empty()) continue;
    data.push_back(image);
    labels.push_back(labelInt);
  }
}
