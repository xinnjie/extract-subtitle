//
// Created by capitalg on 4/11/18.
//

#include <opencv2/imgproc.hpp>
#include "PicClean.h"
#include <iostream>

using namespace cv;
cv::Mat PicClean::keep_white(const cv::Mat &colored_src) {
    Mat gray;
    cv::cvtColor(colored_src, gray, cv::COLOR_RGB2GRAY);

    Mat thres;
    threshold(gray, thres, 215,255,cv::THRESH_BINARY);


    Mat labeled, stats, _centroids;//_centroids unused
    int n = cv::connectedComponentsWithStats(thres, labeled,stats, _centroids, 8);

    // 如果连通体过大、过小，删除该连通体
    clear_large_small_conn(labeled, stats, 5, 400);
    labeled.forEach<int>([](int &pixel, const int *position) {
        pixel = pixel == 0 ? 0 : 65535;
    });

    Mat mask;
    labeled.convertTo(mask, CV_8UC1);
    unsigned char data[3][3] = {
            {0,1,0},
            {1,1,1},
            {0,1,0}
    };
    Mat kernel(3,3, CV_8UC1, &data);
    cv::dilate(mask, mask, kernel);
    Mat clean;
    gray.copyTo(clean, mask);
    return clean;
}

void PicClean::clear_large_small_conn(cv::Mat &labeled, const cv::Mat &stats, int min, int max) {
    for (int label = 0; label < stats.rows; ++label) {
        int area_size = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area_size < min || area_size > max ) {
            int top = stats.at<int>(label, CC_STAT_TOP),
                    left = stats.at<int>(label, CC_STAT_LEFT),
                    width = stats.at<int>(label, CC_STAT_WIDTH),
                    height = stats.at<int>(label, CC_STAT_HEIGHT);
            Mat remove_area = Mat(labeled,cv::Rect(left, top, width, height));
            for (int i = 0; i < remove_area.rows ; ++i) {
                int *rowi = remove_area.ptr<int>(i);
                for (int j = 0; j < remove_area.cols; ++j) {
                    rowi[j] =
                            rowi[j] == label ? 0 : rowi[j];
                }
            }
        }
    }
}
//#include <iostream>
std::pair<int, int> PicClean::locate_subtitle(const cv::Mat &gray_src) {
    Mat hist;
    cv::reduce(gray_src, hist, 0, CV_REDUCE_SUM, CV_32SC1);

    int x1 = 0, x2 = 0;
    // 连续二十列总体亮度都很低表示没有字
    // 字幕开头大约出现在 1/4 处
    for (int i = gray_src.cols / 4; i < gray_src.cols ; ++i) {
        int count = 0;
        while (hist.at<int>(i) < 500 && i < gray_src.cols) {
            ++i;
            count++;
        }
        if (count > 20) {
            x2 = i - count;
            break;
        }
    }

    for (int i = gray_src.cols / 4; i > 0 ; --i) {
        int count = 0;
        while (hist.at<int>(i) < 500 && i > 0) {
            --i;
            count++;
        }
        if (count > 20) {
            x1 = i + count;
            break;
        }
    }
//    std::cout << x1 << "   " << x2 << std::endl;
    return std::make_pair(x1, x2);
}

cv::Mat PicClean::keep_common(std::vector<cv::Mat> srcs) {
    assert(!srcs.empty());
    Mat img = srcs[0];
    for (int i = 0; i < srcs.size(); ++i) {
        Mat diff;
        cv::absdiff(img, srcs[i], diff);
        diff = diff > 200;
//        std::cout << diff.size << std::endl;
//        std::cout << img.size << std::endl;

        img = img - diff.mul(img);
    }
    return img;
}

bool PicClean::same_subtitle(const cv::Mat &src1, const cv::Mat &src2, int thres, int x1, int x2) {
    Rect region(x1, 0, x2-x1, src1.rows);
    const Mat area1 = Mat(src1, region),
        area2 = Mat(src2, region);

    Mat diff;
    cv::absdiff(src1, src2, diff);
//    std::cout << diff.size  << std::endl;

//    Mat sum;
//    reduce(diff, sum, 0, CV_REDUCE_SUM, CV_32SC1);
//    std::cout << sum  << std::endl;
//    std::cout << sum.size  << std::endl;
//
//    reduce(sum, sum, 1, CV_REDUCE_SUM);
//    return sum.at<int>(0)  < thres;
    return cv::sum(diff).val[0]  < thres;
}

bool PicClean::is_blank(const cv::Mat src) {
    auto p = locate_subtitle(src);
    int x1 = p.first,
        x2 = p.second;
    if (x1 == 0 && x2 == 0) {
        return true;
    }
    return sum_mat(src) < 1000;
}

int PicClean::sum_mat(const cv::Mat &src) {
    Mat sum;
    reduce(src, sum, 1, CV_REDUCE_SUM, CV_32SC1);
    reduce(sum, sum, 0, CV_REDUCE_SUM);
    return sum.at<int>(0);
}


