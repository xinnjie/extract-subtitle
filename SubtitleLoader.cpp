//
// Created by capitalg on 4/12/18.
//

#include <string>
#include <iostream>
#include <opencv2/text.hpp>
#include <random>
#include "SubtitleLoader.h"


using namespace std;
using namespace cv;


/*
 * helper functions
 */
int get_random(int lower, int upper);
bool upper_lower_almost_same(const Rect &lh, const Rect &rh);
inline bool almost_same(int lh, int rh, int error=5);
std::pair<int, int> choose_most_frequent_upper_lower_bound(const vector<Rect> &regions);
std::pair<int, int> where_are_subtitles(const vector<Mat> &frames);
/**
 * key function, 接收一个视频，返回字幕的上下边界y坐标
 * @param capt
 * @param sampling_times
 * @return
 */
std::pair<int, int> where_are_subtitles(cv::VideoCapture &capt, int sampling_times=10);

std::vector<cv::Rect> locate_text(Mat &src);



SubtitleLoader::SubtitleLoader(std::string file_path, double interval) : capt(file_path), current_frame(0) {
    if (!capt.isOpened())
    {
        cout  << "Could not open"<< endl;
        exit(-1);
    }
    auto bound = where_are_subtitles(capt);
    this->upper = bound.first;
    this->lower = bound.second;
    this->total_frames_count = (int)capt.get(CAP_PROP_FRAME_COUNT);
    this->video_height = (int)capt.get(CAP_PROP_FRAME_HEIGHT);
    this->video_width = (int)capt.get(CAP_PROP_FRAME_WIDTH);
    auto   frame_rate = (int)capt.get(CAP_PROP_FPS);
    this->interval = (int) (frame_rate * interval);
}

cv::Mat SubtitleLoader::load() {
    Mat frame;
    capt >> frame;
    if (frame.empty()) {
        throw OutOfFrames();
    }
    frame = Mat(frame, Rect(0, this->upper, this->video_width, this->lower - this->upper));
    set_frame_id(get_frame_id()+this->interval);

    return frame;
}

void SubtitleLoader::set_frame_id(int id) {
    this->current_frame = id;
    capt.set(CAP_PROP_POS_FRAMES, this->current_frame);
}

int SubtitleLoader::get_frame_id() {
    return this->current_frame;
}


/**
 * 字幕的特点，在不同的帧中，竖直坐标不变，高度保持一致（一般字体大小不会改变），宽度随字数改变
 * 而 台标类似物 各个参数都不会改变
 * @param regions
 * @return upper bound and lowerbound for the subtitle area
 */
std::pair<int, int> choose_most_frequent_upper_lower_bound(const vector<Rect> &regions) {
    vector<pair<Rect, int>> rect_types;
    for (auto& rect : regions) {
        auto it = std::find_if(rect_types.begin(), rect_types.end(),
                               [&rect](const pair<Rect, int> &item) {
                                   return upper_lower_almost_same(item.first, rect);
                               });
        if (it == rect_types.cend()) {
            rect_types.push_back(make_pair(rect, 0));
        }
    }

    // 找到最常出现的上下坐标
    vector<pair<Rect, int>> counts;
    for (auto &rect: regions) {
        for (auto &type_pair : rect_types) {
            if (upper_lower_almost_same(rect, type_pair.first)) {
                type_pair.second++;
                break;
            }
        }
    }
    auto selected = rect_types[0];
    for (auto &type_pair : rect_types) {
        if (type_pair.second > selected.second) selected = type_pair;
    }
    auto the_rect = selected.first;
    // 上下边界留下可容忍的额外部分
    return make_pair(the_rect.y - 5, the_rect.y + the_rect.height +3 );

}



std::pair<int, int> where_are_subtitles(const vector<Mat> &frames) {
    vector<Rect> regions;
    int height = frames[0].rows,
            width = frames[0].cols;
    for (auto &frame: frames) {
//        imshow("debug", frame);
//        waitKey();
//        // debug
//        auto dup = DetectText::SWT(frame, false);
//        imshow("debug", dup);
//        waitKey();

//        auto rects = DetectText::locate_text(frame, false);
        Mat half(frame, Rect(0, height/2, width, height/2));
        auto rects = locate_text(half);
        for (auto &rect: rects) {
            rect.y += height/2;
        }
//        // debug
//        for (auto &rect : rects) {
//            imshow("debug", Mat(frame, rect));
//        }
        regions.insert(regions.end(), rects.begin(), rects.end());
    }
    return choose_most_frequent_upper_lower_bound(regions);
};


std::pair<int, int> where_are_subtitles(cv::VideoCapture &capt, int sampling_times) {
    vector<Mat> frames;
    Mat frame;
    int frame_id = 0;
    int total_frames_count = capt.get(CAP_PROP_FRAME_COUNT);
//    int height = capt.get(cv::CAP_PROP_FRAME_HEIGHT),
//        width = capt.get(cv::CAP_PROP_FRAME_WIDTH);
    for (int i = 0; i < sampling_times; ++i) {
        frame_id = get_random(0, total_frames_count-1);
        cout << "frame id: " <<  frame_id << endl;
        capt.set(CAP_PROP_POS_FRAMES, frame_id);
        capt >> frame;
        frames.push_back(frame.clone());
    }

    return where_are_subtitles(frames);
};


std::vector<cv::Rect> locate_text(Mat &src) {
//        Mat src = SWT(input, dark_on_light);
//    cv::cvtColor(src, src, cv::COLOR_GRAY2RGB);
    // Extract channels to be processed individually
    std::vector<Mat> channels;

    cv::text::computeNMChannels(src, channels);

    int cn = (int)channels.size();
    // Append negative channels to detect_in_gray ER- (bright regions over dark background)
    for (int c = 0; c < cn-1; c++)
        channels.push_back(255-channels[c]);

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<cv::text::ERFilter> er_filter1 = cv::text::createERFilterNM1(cv::text::loadClassifierNM1("/Users/gexinjie/codes/text_detect/trained_classifierNM1.xml"),16,0.00015f,0.13f,0.2f,true,0.1f);
    Ptr<cv::text::ERFilter> er_filter2 = cv::text::createERFilterNM2(cv::text::loadClassifierNM2("/Users/gexinjie/codes/text_detect/trained_classifierNM2.xml"),0.5);

    std::vector<std::vector<cv::text::ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel (could be done in parallel)
    std::cout << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << std::endl;
    std::cout << "    (...) this may take a while (...)" << std::endl;
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    // Detect character groups
    std::cout << "Grouping extracted ERs ... ";
    std::vector< std::vector<Vec2i> > region_groups;
    std::vector<Rect> groups_boxes;
    erGrouping(src, channels, regions, region_groups, groups_boxes, cv::text::ERGROUPING_ORIENTATION_HORIZ);
    //erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);

    // draw groups
//        groups_draw(src, groups_boxes);
//        imshow("grouping",src);

    std::cout << "Done!" << std::endl;
    // memory keep_white-up
    er_filter1.release();
    er_filter2.release();
    regions.clear();

    std::vector<cv::Rect> unique_boxes;
    for (auto &item : groups_boxes) {
        if (std::find(unique_boxes.cbegin(), unique_boxes.cend(), item) == unique_boxes.cend()) {
            unique_boxes.push_back(item);
        }
    }
//        groups_boxes.erase(std::unique(groups_boxes.begin(), groups_boxes.end()), groups_boxes.end());
    return unique_boxes;
}


bool upper_lower_almost_same(const Rect &lh, const Rect &rh) {
    return almost_same(lh.height + lh.y, rh.height + rh.y) &&
           almost_same(lh.y, rh.y);
}

bool almost_same(int lh, int rh, int error) {
    return abs(lh-rh) <= error;
}


int get_random(int lower, int upper) {
    static std::random_device rd;  //Will be used to obtain a seed for the random number engine
    static std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(lower, upper);
    return dis(gen);
}