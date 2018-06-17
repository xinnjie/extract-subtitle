#include <iostream> // for standard I/O
#include <iomanip>  // for controlling float print precision
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <map>
#include <random>
#include <opencv2/text.hpp>
#include <boost/filesystem.hpp>
#include "TextDetection/TextDetection.h"
#include "SubtitleLoader.h"
#include "PicClean.h"

using namespace std;
using namespace cv;
namespace fs =  boost::filesystem;

int same_thres_per = 1500;
int id = 0;
cv::Size SUBTITLESIZE(850, 35);

Mat process_imgs(vector<Mat> &imgs, string out_path) {
    auto path = fs::system_complete(out_path);
    auto out = PicClean::keep_common(imgs);
//    imshow("out", out);
    imwrite(path.string()+ "/" + to_string(id++) + ".jpg", out);
//    imshow("out", out);
//    waitKey();
    // 每输出10张报告一次
    if (id % 10 == 0) {
        cout << id << " completed" << endl;
    }
    imgs.clear();
    return out;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cerr << "usage: input_video output_dir_name" << endl;
        exit(-1);
    }
    string input_video = argv[1],
        dir_name = argv[2];

    fs::create_directory(dir_name);

    SubtitleLoader lder(input_video, 0.2);
    lder.set_frame_id(0);
    vector<Mat> imgs;
    try {
        while (true) {
            auto frame = lder.load();
//            resize
            cv::resize(frame, frame, SUBTITLESIZE);


            frame = PicClean::keep_white(frame);
//            imshow("frame", frame);
//            waitKey();
            auto bound = PicClean::locate_subtitle(frame);
            if (abs(bound.first - bound.second) < 5) {
                if (!imgs.empty()) {
                    process_imgs(imgs, dir_name);
                }
                else {
                    continue;
                }
            }
            else {
                if (imgs.empty()) {
                    imgs.push_back(frame);
                }
                else {
                    if (PicClean::same_subtitle(imgs.back(), frame, same_thres_per*abs(bound.second-bound.first), bound.first, bound.second)) {
                        imgs.push_back(frame);
                    }
                    else {
                        process_imgs(imgs, dir_name);
                    }
                }
            }

        }
    }
    catch (OutOfFrames e) {
    }

    return 0;
}




