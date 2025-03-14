#include "OrientationDetector.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <cmath>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Constructor that takes the path to a reference image
OrientationDetector::OrientationDetector(const std::string &refImagePath) {
  // Check if the reference image exists
  if (!fs::exists(refImagePath)) {
    cerr << "Error: Reference image '" << refImagePath << "' not found." << endl;
    throw runtime_error("Reference image not found");
  }

  // Load the image in grayscale
  Mat grayRefImage = imread(refImagePath, IMREAD_GRAYSCALE);
  if (grayRefImage.empty()) {
    cerr << "Error: Reference image '" << refImagePath << "' failed to load." << endl;
    throw runtime_error("Reference image load error");
  }

  // For debugging: Compute and print the histogram of pixel intensities
  // const int histSize = 256;
  // float range[] = { 0, 256 };
  // const float* histRange = { range };
  // Mat hist;
  // calcHist(&grayRefImage, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
  
  // cout << "Histogram values:" << endl;
  // for (int i = 0; i < histSize; i++) {
  //   cout << " Intensity " << setw(3) << i << ": " << hist.at<float>(i) << endl;
  // }

  // Create a version of the image where pixel values in the range from 90 to 255 are remapped to the range of 0 to 255
  // Compute scale factor (alpha) and offset (beta)
  double alpha = 255.0 / (255 - 90); // 255/165
  double beta = -90 * alpha;         // Maps 90 -> 0

  // Apply the linear transformation.
  Mat remapped;
  // convertTo automatically saturates values for CV_8U (i.e., values < 0 become 0)
  grayRefImage.convertTo(remapped, CV_8U, alpha, beta);
  Mat lowPassed;
  GaussianBlur(remapped, lowPassed, Size(3, 3), 0);

  // Create a set of rotated reference images.
  // Rotate from 10 to 109 degrees in 1 degree increments.
  // (Using push_back instead of direct indexing.)
  for (int i = 0; i < 200; i++) {
    double angleInDegrees = -100 + i;
    // Compute the rotation matrix using the center of the image.
    Mat rotMatrix = getRotationMatrix2D(Point2f(lowPassed.cols / 2.0, lowPassed.rows / 2.0), -angleInDegrees, 1.0);
    Mat rotated;
    // Apply the rotation
    warpAffine(lowPassed, rotated, rotMatrix, lowPassed.size(), INTER_LINEAR, BORDER_REPLICATE);
    refImages.push_back(rotated);
    imwrite("rotated_pics/rotated_" + std::format("{:03}", i) + ".png", rotated);
  }
}

// Accessor for the rotated reference images
const vector<Mat>& OrientationDetector::getRefImages() const {
  return refImages;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// High-pass filter a 1D signal using FFT.
// 'signal' is the input 1D data, and 'cutoff' is the normalized cutoff frequency (0–1),
// below which frequencies will be removed.
vector<pair<double, double>> highPassFilterFFT(const vector<pair<double, double>>& signal, double cutoff) {
  int N = signal.size();
  
  // Create a 1xN matrix of type CV_64F for the input signal.
  Mat signalMat(1, N, CV_64F);
  for (int i = 0; i < N; i++) {
    signalMat.at<double>(0, i) = signal[i].second;
  }
  
  // Compute the forward DFT. The output is a 1xN matrix of complex numbers (CV_64FC2).
  Mat dftMat;
  dft(signalMat, dftMat, DFT_COMPLEX_OUTPUT);
  
  // The frequency resolution is 1/N (normalized frequency).
  // We zero out the frequency components with index < cutoffIndex.
  int cutoffIndex = static_cast<int>(cutoff * N);
  // For real signals, the DFT is symmetric, so you might want to zero out the corresponding high frequency part too.
  for (int i = 0; i < cutoffIndex; i++) {
    dftMat.at<Vec2d>(0, i)[0] = 0; // Real part
    dftMat.at<Vec2d>(0, i)[1] = 0; // Imaginary part
  }
  // Also clear the symmetric part (for indices near N)
  for (int i = N - cutoffIndex; i < N; i++) {
    dftMat.at<Vec2d>(0, i)[0] = 0;
    dftMat.at<Vec2d>(0, i)[1] = 0;
  }
  
  // Compute the inverse DFT.
  Mat idftMat;
  dft(dftMat, idftMat, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
  
  // Convert the resulting Mat back to a vector<double>.
  vector<pair<double, double>> filteredSignal(N);
  for (int i = 0; i < N; i++) {
    filteredSignal[i].first = signal[i].first;
    filteredSignal[i].second = idftMat.at<double>(0, i);
  }
  
  return filteredSignal;
}

// Helper function for quadratic (parabolic) interpolation.
// For equally spaced measurements, this function finds the vertex of the parabola
// passing through three consecutive points. For maximum correlation, we want the vertex
// corresponding to the highest correlation.
static double parabolicInterpolation(const vector<pair<double, double>> &measurements, double step, double &curvature) {

  // Find the index with the maximum correlation.
  unsigned int bestIdx = 0;
  for (size_t i = 1; i < measurements.size(); i++) {
    if (measurements[i].second > measurements[bestIdx].second)
      bestIdx = i;
  }
  // If the best candidate is at the boundary, return it.
  if (bestIdx == 0 || bestIdx == measurements.size() - 1) {
    curvature = 0;
    return measurements[bestIdx].first;
  }
  
  double y0 = measurements[bestIdx - 1].second;
  double y1 = measurements[bestIdx].second;
  double y2 = measurements[bestIdx + 1].second;
  double denominator = 2 * (y0 - 2 * y1 + y2);
  if (fabs(denominator) < 1e-6) {
    curvature = 0;
    return measurements[bestIdx].first;
  }
  double offset = step * (y0 - y2) / denominator;
  // Compute the quadratic coefficient a (the curvature term).
  curvature = (y0 - 2 * y1 + y2) / (2 * step * step);
  return measurements[bestIdx].first + offset;
}

// This member function searches for the rotation angle that maximizes the correlation
// between the input image ROI and a reference image from refImages. The search is performed
// in two stages (coarse and fine) and quadratic interpolation is used to refine the best angle.
// The computed 'confidence' is based on the x^2 coefficient (curvature) of the parabolic fit.
std::pair<double, double> OrientationDetector::detectRotationAngle(const Mat &frame, Rect roi, const std::string &label, int frame_number) {

  double refinedAngle;
  double confidence;

  for (int y_offset = 0; y_offset<=0; y_offset++) {
    for (int x_offset = 0; x_offset<=0; x_offset++) {

      Rect roiAdjusted = roi;
      roiAdjusted.x += x_offset;
      roiAdjusted.y += y_offset;

      // Extract the ROI from the frame and convert to grayscale.
      Mat cropped = frame(roiAdjusted);
      Mat grayROI, grayROI_f;
      cvtColor(cropped, grayROI, COLOR_BGR2GRAY);
      grayROI.convertTo(grayROI_f, CV_32F);
      
      // For this method, we compare grayROI with our precomputed rotated reference images.
      // We assume refImages are computed from a base angle (e.g., baseAngle = 10 degrees) in 1° increments.
      double baseAngle = -100.0;
      int n = refImages.size();  // e.g., 100
      double curvature = 0.0;

      // bool dumDebugPics = false && frame_number % 500 == 0
      bool dumpDebugPics = false; //(frame_number == 750 || frame_number == 12112 || frame_number == 12103) && (label=="ship_angle");
      
      // Coarse search: use a coarse step (e.g., every 5° i.e. every 5 images).
      vector<pair<double, double>> coarseMeasurements;
      int coarseStep = 4;  // Coarse sampling: step = 5 images (i.e., 5 degrees).
      for (int i = 0; i < n; i += coarseStep) {
        // Get the precomputed reference image.
        Mat ref = refImages[i], ref_f;
        ref.convertTo(ref_f, CV_32F);
        // If necessary, resize to match grayROI.
        if (ref.size() != grayROI.size()) cerr << "Warning: reference image size does not match region." << endl;
        // Compute the correlation metric: sum of pixel-wise products.
        Scalar corrScalar = sum(grayROI_f.mul(ref_f));
        double corr = corrScalar[0];
        coarseMeasurements.push_back({i, corr});
        if (dumpDebugPics) {
          double angleCandidate = baseAngle + i;
          cout << "Coarse angle: " << angleCandidate << " Correlation: " << corr << endl;
          Mat absDiff;
          absdiff(grayROI, ref, absDiff);
          //imwrite("cropped_pics/coarse_" + label + "_" + to_string(frame_number) + "_" + std::format("{:03}", i) + "_" + to_string(corr) + ".png", grayROI_f.mul(ref_f)*0.01);
          imwrite("cropped_pics/" + label + "_" + to_string(frame_number) + "_coarse_" + std::format("{:03}", i) + "_" + to_string(corr) + ".png", absDiff);
        }
      }

      double cutoff = 0.05; // 0.1 normalized cutoff frequency
      vector<pair<double, double>> filteredCoarseMeasurements = highPassFilterFFT(coarseMeasurements, cutoff);

      if (dumpDebugPics) {
        cout << frame_number << endl;
        for (int i = 0; i < n; i += coarseStep) {
          double angleCandidate = baseAngle + i;
          cout << "Filtered coarse angle: " << angleCandidate << " Correlation: " << coarseMeasurements[i/coarseStep].second << " " << filteredCoarseMeasurements[i/coarseStep].second << endl;
        }
      }

      // Use quadratic interpolation on the course measurements.
      double refinedIndex = parabolicInterpolation(filteredCoarseMeasurements, coarseStep, curvature);
      int bestCoarseIndex = (int)round(refinedIndex);
      int coarseConfidence = round(-curvature);
      bool fullFineSearch = (coarseConfidence < 2000);
      
      // Fine search: refine in a narrow range around the best coarse angle.
      vector<pair<double, double>> fineMeasurements;
      int fineStep = 1;  // degrees
      int fineRange = 5; // search +/- 5 degrees around best coarse candidate.
      int fineStart = fullFineSearch ? 0 : max(0, bestCoarseIndex - fineRange);
      int fineEnd = fullFineSearch ? n-1 : min(n-1, bestCoarseIndex + fineRange);
      for (int i = fineStart; i <= fineEnd; i += fineStep) {
        // Get the precomputed reference image.
        Mat ref = refImages[i], ref_f;
        // Compute correlation metric.
        ref.convertTo(ref_f, CV_32F);
        Scalar corrScalar = sum(grayROI_f.mul(ref_f));
        double corr = corrScalar[0];
        fineMeasurements.push_back({i, corr});
        if (dumpDebugPics) {
          double angleCandidate = baseAngle + i;
          cout << "Fine angle: " << angleCandidate << " Correlation: " << corr << endl;
          Mat absDiff;
          absdiff(grayROI, ref, absDiff);
          //imwrite("cropped_pics/fine_" + label + "_" + to_string(frame_number) + "_" + std::format("{:03}", i) + "_" + to_string(corr) + ".png", grayROI_f.mul(ref_f)*0.01);
          imwrite("cropped_pics/" + label + "_" + to_string(frame_number) + "_fine_" + std::format("{:03}", i) + "_" + to_string(x_offset) + "_"  + to_string(y_offset) + "_" + to_string(corr) + ".png", absDiff);
        }
      }
      
      // Use quadratic interpolation on the fine measurements.
      refinedAngle = baseAngle + parabolicInterpolation(fineMeasurements, fineStep, curvature);

      // Assess confidence: if curvature is large and negative, confidence is high.
      // Here, we map a curvature of -10 or less to 100 confidence.

      if (curvature < 0)
        confidence = min(100.0, 100.0 * (-curvature) / 10.0);
      else
        confidence = 0.0;

      if (false && frame_number % 10 == 0) {
        // Add the ref image to the cropped image
        int refIndex = max(0, min(179, (int)round(refinedAngle-baseAngle)));
        Mat ref = refImages[refIndex];
        // Convert the ref image back to RGB
        cvtColor(ref, ref, COLOR_GRAY2BGR);
        if (ref.size() != cropped.size()) cerr << "Warning: reference image size does not match region." << label << " " << to_string(frame_number) << endl;
        else cropped = cropped * 0.75 + ref * 0.25;
        // Draw a line on cropped to represent the detected angle
        Point2f p1(roi.width / 2, roi.height / 2);
        Point2f p2(p1.x + 50 * cos((90-refinedAngle) * CV_PI / 180), p1.y - 50 * sin((90 - refinedAngle) * CV_PI / 180));
        line(cropped, p1, p2, Scalar(0, 0, 255), 1);
        imwrite("cropped_pics/" + label + "_" + to_string(frame_number) + "_" + to_string((int)refinedAngle) + "_" + to_string(coarseConfidence) + ".png", cropped);
      }

          
      // if (frame_number % 500 == 0) {
      //   cout << "Refined angle: " << refinedAngle << " Confidence: " << confidence << endl;
      // }

    }
  }
  return {refinedAngle, confidence};
}