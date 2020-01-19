#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace cv::ml;

using namespace std;

void loadData(string path, vector<Mat> &data, vector<int> &label,
              float labelInt);

int main(int argc, char *argv[]) {
  cv::String trEm("images/training/empty/*.png");
  cv::String trOT("images/training/onTrack/*.png");
  cv::String trT("images/training/train/*.png");

  cv::String tB("images/test/barrier/*.png");
  cv::String tEm("images/test/empty/*.png");
  cv::String tEn("images/test/entering/*.png");
  cv::String tL("images/test/leaving/*.png");
  cv::String tOT("images/test/onTrack/*.png");
  cv::String tT("images/test/train/*.png");

  Mat maskZoneA = imread("images/masks/maskZoneA.png", CV_LOAD_IMAGE_GRAYSCALE),
      gray, maskedImage, reshapedImage;

  vector<int> labels;

  vector<Mat> data, vectorDataTrain;

  Ptr<SVM> svm = SVM::create();

  Mat im, image;
  Mat imTest, matTest;

  int rows, cols;

  loadData(trEm, data, labels, -1);
  loadData(trOT, data, labels, -1);
  loadData(trT, data, labels, 1);

  for (size_t k = 0; k < data.size(); ++k) {
    cvtColor(data[k], gray, CV_BGR2GRAY);
    gray.copyTo(maskedImage, maskZoneA);
    reshapedImage = maskedImage.reshape(1, 1);
    reshapedImage.convertTo(reshapedImage, CV_32F);
    vectorDataTrain.push_back(reshapedImage);
  }

  rows = labels.size();
  cols = reshapedImage.cols;

  // Set up training data
  Mat labelSetTrain(labels, false);
  Mat trainingDataSetTrain(rows, cols, CV_32F);

  for (size_t i = 0; i < rows; i++) {
    vectorDataTrain[i].copyTo(trainingDataSetTrain(Rect(0, i, cols, 1)));
  }

  // Set up SVM's parameters
  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::RBF);

  // Train the SVM
  svm->trainAuto(
      TrainData::create(trainingDataSetTrain, ml::ROW_SAMPLE, labelSetTrain));
  svm->save("Models/trainModel.xml");
}

void loadData(string path, vector<Mat> &data, vector<int> &labels,
              float labelInt) {
  vector<cv::String> fn;
  Mat image;

  cv::glob(path, fn, false);

  for (size_t k = 0; k < fn.size(); ++k) {
    image = cv::imread(fn[k]);
    if (image.empty()) continue;
    data.push_back(image);
    labels.push_back(labelInt);
  }
}
