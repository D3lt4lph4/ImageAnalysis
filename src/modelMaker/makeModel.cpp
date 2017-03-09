/*
* This program allow the user to create the model for the SVM for the event detection of the second program.
*
* To use it : programName
*
* WARNING :
* Be aware that the program needs to access images/files under the following pathes : Images/path/to/image Models/path/to/model.
* Thus the corresponding directories need to be placed at the correct location.
*
* This program was made using the lectures notes from Image Analysis, the provided files and the opencv documentation and examples.
*
* \author Benjamin Deguerre
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>
#include <filesystem>

#define PI 3.14159265358979323846

using namespace cv;
using namespace std;
using namespace std::tr2::sys;

void loadData(string path, vector<Mat> &data, vector<float> &label, float labelInt);

int main(int argc, char *argv[]) {

  cv::String trEm("Images/Training/Empty");
  cv::String trOT("Images/Training/OnTrack");
  cv::String trT("Images/Training/Train");

  Mat maskZoneA = imread("Images/Masks/maskZoneA.png",CV_LOAD_IMAGE_GRAYSCALE), gray, maskedImage, reshapedImage;
  vector<float> labels;

  vector<Mat> data, vectorDataTrain, vectorDataEmpty;

  CvSVM SVM;
  CvSVMParams params;

  Mat im , image,canny_output, original, yolo, yolo2,dst;
  Mat imTest, matTest;

  int rows, cols, thresh = 75;

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

  for (int i = 0; i < rows; i++) {
    vectorDataTrain[i].copyTo(trainingDataSetTrain(Rect(0,i,cols,1)));
  }

  // Set up SVM's parameters
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;

  // Train the SVM
  SVM.train_auto(trainingDataSetTrain, labelSetTrain, Mat(), Mat(), params);
  SVM.save("Models/trainModel.xml");
}

void loadData(string path, vector<Mat> &data, vector<float> &labels, float labelInt) {

  vector<cv::String> fn;
  Mat image;

  for (recursive_directory_iterator i(path), end; i != end; ++i) {
    if (!is_directory(i->path())) {
      image = cv::imread(i->path().string());
      if (image.empty()) continue;
      data.push_back(image);
      labels.push_back(labelInt);
    }
  }
}
