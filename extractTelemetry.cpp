#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <json/json.h>
#include <algorithm>

using namespace cv;
using namespace std;

struct OCRRegion {
  string label;
  Rect boundingBox;
};

// Define regions for extraction
vector<OCRRegion> regions = {
  {"timer", Rect(856, 946, 206, 39)},
  {"boost_speed", Rect(333, 912, 113, 29)},
  {"boost_alt", Rect(362, 948, 88, 26)},
  {"ship_speed", Rect(1518, 912, 113, 29)},
  {"ship_alt", Rect(1538, 948, 93, 26)}
};

// Function to extract text from an image and report confidence level
pair<string, float> extractTextWithConfidence(Mat &frame, Rect roi, tesseract::TessBaseAPI &ocr, bool isTime = false) {
  Mat cropped = frame(roi);
  Mat gray;
  cvtColor(cropped, gray, COLOR_BGR2GRAY);
  threshold(gray, gray, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

  // Set the image for OCR
  ocr.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
  ocr.SetSourceResolution(300);  // Ensure a set DPI

  // Get text
  string text = ocr.GetUTF8Text();

  // Get confidence
  float confidence = -1.0f;  // Default in case no confidence is found
  tesseract::ResultIterator* ri = ocr.GetIterator();
  if (ri) {
  confidence = ri->Confidence(tesseract::RIL_TEXTLINE);  // Confidence at line level
  delete ri;  // Avoid memory leak
  }

  // Clean text output
  text.erase(remove_if(text.begin(), text.end(), ::isspace), text.end());
  if (!isTime) {
  text.erase(remove_if(text.begin(), text.end(), [](char c) { return !isdigit(c); }), text.end());
  }

  return {text, confidence};  // Return both text and confidence
}

int main() {
  string video_path = "StarshipIFT7.mp4"; // Change as needed
  string output_js_file = "StarshipIFT7.js";

  VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    cerr << "Error: Could not open video file." << endl;
    return -1;
  }

  ofstream jsFile(output_js_file);
  jsFile << "export const StarshipIFT7 = [\n";

  tesseract::TessBaseAPI ocr;
  if (ocr.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
    cerr << "Could not initialize Tesseract OCR." << endl;
    return -1;
  }
  ocr.SetVariable("tessedit_char_whitelist", "0123456789T+-:");

  bool liftoff = false;
  bool alreadyPrintedHeadings = false;
  int startFrame = 0;
  int frame_count = 0;
  int num_frames = 1000;

  while (true || (frame_count < num_frames)) {
    Mat frame;
    if (!cap.read(frame)) {
      cout << "End of video or error reading frame." << endl;
      break;
    }

    Json::Value extracted_data;
    extracted_data["frame"] = frame_count;

    for (const auto &region : regions) {
      auto [text, confidence] = extractTextWithConfidence(frame, region.boundingBox, ocr, region.label == "timer");

      if (region.label == "timer") {
        extracted_data["timer"] = text;
        extracted_data["timer_confidence"] = confidence;
        if (!liftoff && text.find("T+") == 0) {
          cout << "Liftoff detected!" << endl;
          liftoff = true;
          startFrame = frame_count;
        }
      } else if (liftoff) {
        try {
          extracted_data[region.label] = stoi(text);
        } catch (...) {
          extracted_data[region.label] = "NaN";
        }
        extracted_data[region.label+"_confidence"] = confidence;
      }
    }

    if (liftoff) {
      extracted_data["timeInSec"] = max(0.0, round((frame_count - startFrame) * 1024 / 30) / 1024);
      jsFile << extracted_data.toStyledString() << ",\n";
      if (!alreadyPrintedHeadings) {
        // Print out headings just once
        cout << setw(20) << " ";
        for (const auto &region : regions) {
          cout << setw(10) << region.label.substr(0, 10) << " ";  // Print shortened key names
        }
        cout << "(Conf: ";
        for (const auto &region : regions) {
          cout << setw(4) << region.label.substr(0, 4) << " ";
        }
        cout << ")" << endl << flush;
        alreadyPrintedHeadings = true;
      }
    }


    // Print extracted values (excluding confidence values)
    cout << "\rFrame: " << setw(6) << frame_count << " Data: ";

    for (const auto &region : regions) {
      Json::Value value = extracted_data[region.label];  // Correctly retrieve the value
      
      if (value.isNumeric())  // Check if the value is a valid number
        cout << setw(10) << value.asInt() << " ";  // Fixed width, right-aligned
      else
        cout << setw(10) << value.asString() << " ";  // Print as string (handles "NaN")
    }
    
    // Print confidence values separately
    cout << "(Conf:";
    for (const auto &region : regions) {
      string confidenceKey = region.label + "_confidence";
      
      if (extracted_data.isMember(confidenceKey)) {  // Check if confidence value exists
        Json::Value confidenceValue = extracted_data[confidenceKey];
        cout << " " << setw(3) << confidenceValue.asInt() << "%";  // Fixed width for percentages
      } else {
        cout << " --%";  // Placeholder if confidence value is missing
      }
    }
    cout << ") " << flush;
        
    //cout << extracted_data.toStyledString();
    frame_count++;
  }

  jsFile << "];\n";
  jsFile.close();
  cap.release();
  ocr.End();

  cout << "Saved extracted data to " << output_js_file << endl;
  return 0;
}
