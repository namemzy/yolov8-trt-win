#include <stdio.h>
#include <iostream>
#include <fstream>

#include "chrono"
#include "common.h"
#include "yolov8_batch.hpp"


int main()
{
    cudaSetDevice(0);
    int img_h = 640;
    int img_w = 640;
    int batch_size = 4;
    const std::string engine_file_path = "models\\yolov8n_b4.bin";
    std::string out_path = "results\\";

    // 初始化图片列表
    std::vector<std::string> img_path_list;
    img_path_list.push_back("images\\1.jpg");
    img_path_list.push_back("images\\2.jpg");
    img_path_list.push_back("images\\3.jpg");
    img_path_list.push_back("images\\4.jpg");

    auto yolov8 = new YOLOv8(engine_file_path, batch_size);
    yolov8->make_pipe(true);

    cv::Mat res, image_bgr, image;
    cv::Size img_size = cv::Size{ img_w, img_h };
    std::vector<std::vector<Object>> batch_objs;
    std::vector<cv::Mat> images;
    for (auto& path : img_path_list) {
        image_bgr = cv::imread(path);
        image = image_bgr;
        //cv::cvtColor(image_bgr, image, cv::COLOR_BGRA2RGB);
        images.push_back(image);

    }
    batch_objs.clear();
    yolov8->copy_from_Mat(images, img_size);

    auto start = std::chrono::system_clock::now();
    yolov8->infer();
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast <std::chrono::microseconds>(end - start).count() / 1000.;
    std::cout << "cost " << tc << "ms" << std::endl;

    yolov8->postprocess(batch_objs);

    // 保存到本地可视化
    for (int img_id = 0; img_id < img_path_list.size(); img_id++) {
        image_bgr = cv::imread(img_path_list[img_id]);
        yolov8->draw_objects(image_bgr, res, batch_objs[img_id], CLASS_NAMES, COLORS);
        cv::imwrite(out_path + std::to_string(img_id + 1) + ".jpg", res);
    }

    delete yolov8;

    return 0;

}