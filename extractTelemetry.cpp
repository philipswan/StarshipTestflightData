#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <sstream>
#include <vector>
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
  ocr.SetSourceResolution(300);

  // Get text
  string text = ocr.GetUTF8Text();

  // Get confidence
  float confidence = -1.0f;
  tesseract::ResultIterator* ri = ocr.GetIterator();
  if (ri) {
    confidence = ri->Confidence(tesseract::RIL_TEXTLINE);
    delete ri;
  }

  // Clean text output
  text.erase(remove_if(text.begin(), text.end(), ::isspace), text.end());
  if (!isTime) {
    text.erase(remove_if(text.begin(), text.end(), [](char c) { return !isdigit(c); }), text.end());
  }

  return {text, confidence};
}

double timeToFloat(const std::string& timeStr) {
  std::istringstream ss(timeStr);
  std::vector<std::string> parts;
  std::string segment;

  // Split the input string by ':'
  while (std::getline(ss, segment, ':')) {
    parts.push_back(segment);
  }

  // Convert components to seconds
  double totalSeconds = 0.0;
  if (parts.size() == 3) {  // HH:MM:SS
    totalSeconds = std::stod(parts[0]) * 3600 + std::stod(parts[1]) * 60 + std::stod(parts[2]);
  } else if (parts.size() == 2) {  // MM:SS
    totalSeconds = std::stod(parts[0]) * 60 + std::stod(parts[1]);
  } else if (parts.size() == 1) {  // SS
    totalSeconds = std::stod(parts[0]);
  } else {
    throw std::invalid_argument("Invalid time format");
  }

  return totalSeconds;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <video_path> [-ss start_time] [-to end_time]" << endl;
    return 1;
  }

  string video_path;
  double start_time = 0.0;
  double end_time = -1.0;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-ss" && i + 1 < argc) {
      start_time = timeToFloat(argv[i + 1]);
      i++;
    } else if (arg == "-to" && i + 1 < argc) {
      end_time = timeToFloat(argv[i + 1]);
      i++;
    } else if (video_path.empty()) {
      video_path = arg;
    }
  }
  cout << "StartTime:" << start_time << "s EndTime:" << end_time << "s" << endl;

  string dataset_name = video_path.substr(0, video_path.find_last_of('.'));
  string output_js_file = dataset_name + ".js";
  
  VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    cerr << "Error: Could not open video file." << endl;
    return -1;
  }

  double fps = cap.get(CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

  int start_frame = static_cast<int>(start_time * fps);
  int end_frame = (end_time > 0) ? static_cast<int>(end_time * fps) : total_frames - 1;

  if (start_frame >= total_frames) {
    cerr << "Error: Start time exceeds video duration." << endl;
    return 1;
  }
  if (end_frame > total_frames) {
    end_frame = total_frames - 1;
  }

  cap.set(CAP_PROP_POS_FRAMES, start_frame);

  ofstream jsFile(output_js_file);
  jsFile << "export const " << dataset_name << " = [\n";

  tesseract::TessBaseAPI ocr;
  if (ocr.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
    cerr << "Could not initialize Tesseract OCR." << endl;
    return -1;
  }
  ocr.SetVariable("tessedit_char_whitelist", "0123456789T+-:");

  bool liftoff = false;
  bool alreadyPrintedHeadings = false;
  int timerStartFrame = 0;
  int frame_count = start_frame;

  while (frame_count <= end_frame) {
    Mat frame;
    if (!cap.read(frame)) {
      cout << "End of video detected or error reading frame." << endl;
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
          timerStartFrame = frame_count;
        }
      } else if (liftoff) {
        try {
          extracted_data[region.label] = stoi(text);
        } catch (...) {
          extracted_data[region.label] = "NaN";
        }
        extracted_data[region.label + "_confidence"] = confidence;
      }
    }

    if (liftoff) {
      extracted_data["timeInSec"] = max(0.0, round((frame_count - timerStartFrame) * 1024 / 30) / 1024);
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

    frame_count++;
  }

  jsFile << "];\n";
  jsFile.close();
  cap.release();
  ocr.End();

  cout << "Saved extracted data to " << output_js_file << endl;
  return 0;
}
