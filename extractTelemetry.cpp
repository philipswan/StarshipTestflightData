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
#include <sys/stat.h>
#include <sys/types.h>
#include <variant>
#include <optional>
#include <format>
#include <iomanip>
#include <cmath>

constexpr int PRECISION = 2;

using namespace cv;
using namespace std;

struct OCRRegion {
  string label;
  Rect boundingBox;
};
struct verification_check {
  int frame_count;
  string label;
  std::string expected_value;
};

// Define regions for extraction
vector<OCRRegion> regions = {
  {"timer", Rect(856, 946, 206, 39)},
  {"boost_speed", Rect(333, 912, 113, 29)},
  {"boost_alt", Rect(362, 948, 88, 26)},
  {"ship_speed", Rect(1518, 912, 113, 29)},
  {"ship_alt", Rect(1538, 948, 93, 26)},
  {"boost_lox", Rect(275, 1006, 240, 4)},
  {"boost_ch4", Rect(275, 1040, 240, 4)},
  {"ship_lox", Rect(1455, 1005, 240, 4)},
  {"ship_ch4", Rect(1455, 1035, 240, 4)}
};

// Function to compare std::variant with a generic value
std::string floatToStringWithPrecision(float value) {
  return std::format("{:.{}f}", value, PRECISION);
  // For older compilers...
  // std::ostringstream out;
  // out << std::fixed << std::setprecision(PRECISION) << value;
  // return out.str();
}

std::string getFormattedJsonValue(const Json::Value& value) {
  if (value.isDouble()) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(PRECISION) << value.asDouble();
    return out.str();
  }
  return value.asString();  // Default conversion for non-floats
}

// Function to format float values manually
std::string formatFixedPoint(double value) {
  long long scaled = std::llround(value * std::pow(10, PRECISION));
  std::string str_value = std::to_string(std::abs(scaled));  // Convert to string

  // Ensure at least two decimal places exist
  while (str_value.length() < 3) {
    str_value = "0" + str_value;  // Pad with leading zeros if needed
  }

  // Insert decimal point
  std::string formatted = (scaled < 0 ? "-" : "") + str_value.substr(0, str_value.length() - 2) + "." + str_value.substr(str_value.length() - 2);
  
  return formatted;
}

// Function to write JSON with manually formatted floats
void writeJsonWithPrecision(const Json::Value& json, std::ofstream& jsFile) {
  jsFile << "{\n";
  bool first = true;

  for (const auto& key : json.getMemberNames()) {
    if (!first) jsFile << ",\n";  // Proper JSON formatting
    first = false;

    jsFile << "  \"" << key << "\": ";

    // Check if value is a float/double
    if (json[key].isDouble()) {
      jsFile << formatFixedPoint(json[key].asDouble());  // Write manually formatted float
    } 
    // Handle strings properly
    else if (json[key].isString()) {
      jsFile << "\"" << json[key].asString() << "\"";
    } 
    // Handle other data types normally
    else {
      jsFile << json[key];
    }
  }

  jsFile << "\n}";  // Close JSON object
}

// Function to extract text from an image and report confidence level
pair<string, float> extractTextWithConfidence(
  Mat &frame,
  Rect roi,
  tesseract::TessBaseAPI &ocr,
  bool isTime = false,
  std::optional<verification_check> check = std::nullopt)
{
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

  string current_value;
  if (check->label == "timer") {
    current_value = text;
  } else {
    try {
      current_value = to_string(stoi(text)) + "." + std::string(PRECISION, '0');
    } catch (...) {
      current_value = "NaN";
    }
  }
  //if (check) cout << check->label << " " << check->expected_value << " " << current_value << endl;

  // Save the cropped image
  if (check && (check->expected_value != current_value)) {
    cout << "Mismatch: " << check->label << " " << check->expected_value << " " << current_value << endl; 
    Rect extraROI = roi;
    extraROI.y -= 6;
    extraROI.height += 12;
    extraROI.x -= 6;
    extraROI.width += 12;
    Mat cropped = frame(extraROI);
    std::string confidence_string = floatToStringWithPrecision(confidence);
    imwrite("cropped_pics/crop_" + check->label + "_" + to_string(check->frame_count) + "_" + text + "_" + confidence_string + ".png", cropped);
    imwrite("cropped_pics/gray_" + check->label + "_" + to_string(check->frame_count) + "_" + text + "_" + confidence_string + ".png", gray);
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

// Function to calculate fill fraction and confidence of a bar
pair<float, float> extractFillFraction(
  Mat &frame,
  Rect roi,
  const string &label,
  int frame_count,
  std::optional<verification_check> check = std::nullopt)
{

  Mat cropped = frame(roi);

  Mat gray;
  cvtColor(cropped, gray, COLOR_BGR2GRAY);
  threshold(gray, gray, 56, 255, THRESH_BINARY);

  // Save the cropped image
  if (frame_count == 5950) {
    imwrite("grayscale_" + label + "_" + to_string(frame_count) + ".png", gray);
  }
    
  int white_pixels = countNonZero(gray);
  int total_pixels = gray.total();
  float fill_fraction = (total_pixels > 0) ? (white_pixels / static_cast<float>(total_pixels)) : 0;
  int bar_white_width = (int)round(gray.cols * fill_fraction);
  int bar_black_width = gray.cols - bar_white_width;
  Rect roi_white = Rect(0, 0, bar_white_width, gray.rows);
  Rect roi_black = Rect(bar_white_width, 0, bar_black_width, gray.rows);
  Mat bar_white = gray(roi_white);
  Mat bar_black = gray(roi_black);
  int white_pixels_in_white = countNonZero(bar_white);
  int white_pixels_in_black = countNonZero(bar_black);
  int black_pixels_in_white = bar_white.total() - white_pixels_in_white;
  int black_pixels_in_black = bar_black.total() - white_pixels_in_black;
  float confidence = 100.0 * (white_pixels_in_white + black_pixels_in_black) / total_pixels;

  std::string fill_fraction_string = floatToStringWithPrecision(fill_fraction);
  std::string confidence_string = floatToStringWithPrecision(confidence);

  // Save the cropped image
  if (check && (check->expected_value != fill_fraction_string)) {
    Rect extraROI = roi;
    extraROI.y -= 6;
    extraROI.height += 12;
    extraROI.x -= 6;
    extraROI.width += 12;
    Mat cropped = frame(extraROI);
    imwrite("cropped_pics/crop_" + check->label + "_" + to_string(check->frame_count)+ "_" + fill_fraction_string + "_" + confidence_string + ".png", cropped);
    imwrite("cropped_pics/gray_" + check->label + "_" + to_string(check->frame_count)+ "_" + fill_fraction_string + "_" + confidence_string + ".png", gray);
    //exit(0);
  }

  return {fill_fraction, confidence};
}

Json::Value readEditedDataset(const string &filename) {
  ifstream file(filename);
  if (!file) {
    cerr << "Warning: Edited dataset file '" << filename << "' not found. Skipping comparison." << endl;
    return Json::Value();  // Return an empty JSON object
  }

  stringstream buffer;
  buffer << file.rdbuf();
  string jsonContent = buffer.str();

  // Find the first bracket `[`, which marks the start of valid JSON
  size_t bracketPos = jsonContent.find('[');
  if (bracketPos == string::npos) {
    cerr << "Error: JSON structure not found in '" << filename << "'." << endl;
    return Json::Value();
  }

  // Extract only the valid JSON part
  jsonContent = jsonContent.substr(bracketPos);

  // Parse the cleaned JSON
  Json::Value data;
  Json::CharReaderBuilder reader;
  string errors;
  stringstream jsonStream(jsonContent);
  if (!Json::parseFromStream(reader, jsonStream, &data, &errors)) {
    cerr << "Error parsing JSON in '" << filename << "': " << errors << endl;
    return Json::Value();
  }

  return data;
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <video_path> [-ss start_time] [-to end_time] [-dump_cropped_pics]" << endl;
    cerr << "To use -dump_cropped_pics, make a copy of the output file with an \"_edited\" suffix (for example, \"starshipIFT7_edited.js\") and edit some of values in it that appear to be incorrect." << endl;
    cerr << "This will cause the program to save cropped images for the incorrect values in a \"cropped_pics\" subfolder." << endl;
    return 1;
  }
  
  string video_path;
  double start_time = 0.0;
  double end_time = -1.0;
  bool dump_cropped_pics = false;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-ss" && i + 1 < argc) {
      start_time = timeToFloat(argv[i + 1]);
      i++;
    } 
    else if (arg == "-to" && i + 1 < argc) {
      end_time = timeToFloat(argv[i + 1]);
      i++;
    } 
    else if (arg == "-dump_cropped_pics") {
      dump_cropped_pics = true;
      mkdir("cropped_pics", 0777);
    } 
    else if (video_path.empty()) {
      video_path = arg;
    }
  }
  cout << "StartTime:" << start_time << "s EndTime:" << end_time << "s" << endl;

  string dataset_name = video_path.substr(0, video_path.find_last_of('.'));
  string edited_js_file = dataset_name + "_edited.js";
  string output_js_file = dataset_name + ".js";

  Json::Value edited_data;
  if (dump_cropped_pics) {
    // This is a file that was previously generated by this program but subsequently was manually edited to make some corrections.
    edited_data = readEditedDataset(edited_js_file);
  }

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
  jsFile << setprecision(2);

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
    Json::Value edited_entry = Json::nullValue;
    if (liftoff && dump_cropped_pics) {
      edited_entry = edited_data[frame_count-timerStartFrame];
      int frame_count_in_entry = edited_entry["frame"].asInt();
      if (frame_count_in_entry != frame_count) {
        cerr << "Error: Frame count mismatch in edited dataset. " << frame_count_in_entry << " " << frame_count << endl;
        exit(1);
      }

    }

    extracted_data["frame"] = frame_count;

    for (const auto &region : regions) {
      optional<verification_check> check = (liftoff && dump_cropped_pics && !edited_entry.isNull())
        ? optional<verification_check>({frame_count, region.label, getFormattedJsonValue(edited_entry[region.label])})
        : nullopt;

      if (region.label == "timer") {
        auto [text, confidence] = extractTextWithConfidence(frame, region.boundingBox, ocr, region.label == "timer", check);
        extracted_data["timer"] = text;
        extracted_data["timer_conf"] = confidence;
        if (!liftoff && text.find("T+") == 0) {
          cout << "Liftoff detected!" << endl;
          liftoff = true;
          timerStartFrame = frame_count;
          // Grab the first edited entry for the timer
          optional<verification_check> check = (liftoff && dump_cropped_pics && !edited_entry.isNull())
          ? optional<verification_check>({frame_count, region.label, getFormattedJsonValue(edited_entry[region.label])})
          : nullopt;
        }
      } 
      else if (liftoff) {
        if (region.label.find("lox") != string::npos || region.label.find("ch4") != string::npos) {
          auto [fill_fraction, confidence] = extractFillFraction(frame, region.boundingBox, region.label, frame_count, check);
          extracted_data[region.label] = fill_fraction;
          extracted_data[region.label + "_conf"] = confidence;
        }
        else {
          auto [text, confidence] = extractTextWithConfidence(frame, region.boundingBox, ocr, region.label == "timer", check);
          try {
            extracted_data[region.label] = stoi(text);
          } catch (...) {
            extracted_data[region.label] = "NaN";
          }
          extracted_data[region.label + "_conf"] = confidence;
        }
      }
    }

    if (liftoff) {
      extracted_data["timeInSec"] = max(0.0, round((frame_count - timerStartFrame) * 1024 / 30) / 1024);
      writeJsonWithPrecision(extracted_data, jsFile);
      if (frame_count < end_frame) {
        jsFile << ",\n";
      }
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


      //if (frame_count == 5950) exit(0);


    }
    // Print extracted values (excluding confidence values)
    cout << "\rFrame: " << setw(6) << frame_count << " Data: ";

    for (const auto &region : regions) {
      Json::Value value = extracted_data[region.label];  // Correctly retrieve the value
      
      if (value.isNumeric())  // Check if the value is a valid number
        if (region.label.find("lox") != string::npos || region.label.find("ch4") != string::npos)
          cout << setw(10) << fixed << setprecision(2) << value.asDouble() << " ";  // Fixed width, right-aligned
        else
          cout << setw(10) << value.asInt() << " ";  // Fixed width, right-aligned
      else
        cout << setw(10) << value.asString() << " ";  // Print as string (handles "NaN")
    }
    
    // Print confidence values separately
    cout << "(Conf:";
    for (const auto &region : regions) {
      string confidenceKey = region.label + "_conf";
      
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
