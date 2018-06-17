//
// Created by capitalg on 4/11/18.
//

#ifndef TEXT_DETECT_PICCLEAN_H
#define TEXT_DETECT_PICCLEAN_H
#include <opencv2/core.hpp>
#include <utility>

class PicClean {

public:
    /**
     * 保留所有白色区域，
     * 对于白色的字幕，可以基本留下所有字幕部分，但是还会残留白色的背景
     * @param colored_src
     * @return
     */
    static cv::Mat keep_white(const cv::Mat &colored_src);

    /**
     * 保留图像之间一样的部分，
     * 对于相同的字幕，位置不会改变，会被保留。背景如果改变，就会被去除
     * @param srcs
     * @return
     */
    static cv::Mat keep_common(std::vector<cv::Mat> srcs);

    /**
     * 判断是否是同一张字幕图片
     * 具体实现是使用 cv::absdiff(), 若没有超过一定阈值就认定为相同的照片
/////     * 超过阈值也不一定是不同的字幕图片，比如某一帧字幕后面出现大片色块, 这时尝试比对字幕开始、结束地方是否一致，如果一致，认定为同一个字幕
     * @param src1
     * @param src2
     * @param thres   typical 50000
     * @param x1
     * @param x2
     * @return
     */
    static bool same_subtitle(const cv::Mat &src1, const cv::Mat &src2, int thres, int x1, int x2);

    static bool is_blank(const cv::Mat src);


    static std::pair<int, int> locate_subtitle(const cv::Mat &gray_src);
private:
    /**
     * helper function, remove small and large connected components
     * @param labeled label Mat, output of cv::connectedComponentsWithStats()
     * @param stats   stats Mat, output of cv::connectedComponentsWithStats()
     * @param min minimum(exclude) pixels a component should include
     * @param max maximum(exclude) pixels a component should include
     */
    static void clear_large_small_conn(cv::Mat &labeled, const cv::Mat &stats, int min, int max);


    static int sum_mat(const cv::Mat &src);

};


#endif //TEXT_DETECT_PICCLEAN_H
