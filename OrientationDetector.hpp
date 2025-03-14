#ifndef ORIENTATIONDETECTOR_H
#define ORIENTATIONDETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// OrientationDetector is a class that loads a reference image, 
// computes its histogram, and generates a set of rotated reference images.
class OrientationDetector {
public:
  // Constructor that takes the path to a reference image.
  // Throws std::runtime_error if the image cannot be loaded.
  OrientationDetector(const std::string &refImagePath);

  // Detect rotation angle from a given ROI of an input frame.
  // 'confidence' is set based on the curvature of the fitted parabola.
  std::pair<double, double> detectRotationAngle(const cv::Mat &frame, cv::Rect roi, const std::string &label, int frame_number);

  // Returns a const reference to the vector of rotated reference images.
  const std::vector<cv::Mat>& getRefImages() const;

private:
  std::vector<cv::Mat> refImages;
};

#endif // ORIENTATIONDETECTOR_H
