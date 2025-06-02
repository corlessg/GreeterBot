#include<opencv2/opencv.hpp>
#include<iostream>


bool isWaving(const std::deque<int>& x_positions) {
	if (x_positions.size() < 10) return false;

	int minX = *std::min_element(x_positions.begin(), x_positions.end());
	int maxX = *std::max_element(x_positions.begin(), x_positions.end());

	return (maxX - minX > 100); // adjust threshold based on 
}

void model_testing(const cv::Mat& frame) {
	try {
		cv::dnn::Net net = cv::dnn::readNetFromONNX("E:/documents/projects/c++/slnGreeterBot/GreeterBot/models/MediaPipeHandDetector.onnx");

		cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640));
		net.setInput(blob);
		cv::Mat output = net.forward();

		cv::imshow("test circles", frame);

		const int w = frame.cols;
		const int h = frame.rows;

		cv::Mat flat = output.reshape(1, 1);

		float x, y, z;
		int x_px, y_py;;

		//parse each of the values coming out from the output reshaped
		for (int i = 0; i < flat.cols; i += 3) {
			x = flat.at<float>(0, i);
			y = flat.at<float>(0, i);
			z = flat.at<float>(0, i);

			x_px = static_cast<int>(x * w);
			y_py = static_cast<int>(y * h);

			cv::circle(frame, cv::Point(x_px, y_py), 3, cv::Scalar(0, 255, 0), -1);
		}

		cv::imshow("test circles",frame);
		

		x = 0;



	}
	catch (const cv::Exception& e) {
		std::cerr << "OpenCV exception: " << e.what() << "\n";
	}
}

void run_camera() {

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

		/*cv::imshow("Grayscale", gray);
		cv::waitKey(5);*/

		cv::GaussianBlur(gray, gray, cv::Size(21, 21), 0);

		/*cv::imshow("guass", gray);
		cv::waitKey(5);*/

		if (!prevGray.empty()) {
			cv::absdiff(prevGray, gray, diff);
			/*cv::imshow("Diff", diff);
			cv::waitKey(5);*/

			cv::threshold(diff, thresh, 25, 255, cv::THRESH_BINARY);
			/*cv::imshow("Thresh", thresh);
			cv::waitKey(5);*/
			cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);
			/*cv::imshow("Dilate", thresh);
			cv::waitKey(5);*/

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			model_testing(frame);

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

	cv::Mat test_frame;

	test_frame = cv::imread("../GreeterBot/data/WIN_20250601_22_14_41_Pro.jpg");
	while (true) {
		cv::imshow("test", test_frame);
		model_testing(test_frame);
		if (cv::waitKey(1) == 'q') break;
	}
	



	return 0;
}
