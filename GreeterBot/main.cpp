#include<opencv2/opencv.hpp>
#include<iostream>


bool isWaving(const std::deque<int>& x_positions) {
	if (x_positions.size() < 10) return false;

	int minX = *std::min_element(x_positions.begin(), x_positions.end());
	int maxX = *std::max_element(x_positions.begin(), x_positions.end());

	return (maxX - minX > 100); // adjust threshold based on 
}




int main() {

	/*for (int i = 0; i < 10; ++i) {
		cv::VideoCapture cap(i, cv::CAP_DSHOW);
		if (cap.isOpened()) {
			std::cout << "Found it: " << i << "\n";
			cap.release();
		}
		else {
			std::cout << "Not this one: " << i << "\n";
		}
	}*/

	// cap(0) for integrated webcam
	// cap(1+) for 
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Error: cannot open webcam";
	}

	cv::Mat frame, gray, prevGray, diff, thresh;
	std::deque<int> x_positions;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		cv::imshow("Grayscale", gray);
		cv::waitKey(5);

		cv::GaussianBlur(gray, gray, cv::Size(21, 21), 0);

		cv::imshow("guass", gray);
		cv::waitKey(5);

		if (!prevGray.empty()) {
			cv::absdiff(prevGray, gray, diff);
			cv::imshow("Diff", diff);
			cv::waitKey(5);

			cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);
			cv::imshow("Thresh", thresh);
			cv::waitKey(5);
			cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);
			cv::imshow("Dilate", thresh);
			cv::waitKey(5);
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


			for (const auto& contour : contours) {
				if (cv::contourArea(contour) < 1000) continue;

				cv::Rect bound = cv::boundingRect(contour);
				cv::rectangle(frame, bound, cv::Scalar(0, 255, 0), 2);

				int centerX = bound.x + bound.width / 2;
				x_positions.push_back(centerX);
				if (x_positions.size() > 20) x_positions.pop_front();

				if (isWaving(x_positions)) {
					cv::putText(frame, "Waving Detected!", { 50, 50 },
						cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 0, 255 }, 2);
				}
			}
		}

		prevGray = gray.clone();
		cv::imshow("Waving Detection", frame);
		if (cv::waitKey(1) == 'q') break;
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}
