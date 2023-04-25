
#include "fstream"
#include "NvInferPlugin.h"
using namespace det;

class YOLOv8
{
public:
    explicit YOLOv8(const std::string& engine_file_path, int batch_size);
    ~YOLOv8();

    void make_pipe(bool warmup = true);
    void copy_from_Mat(const std::vector<cv::Mat>& images, cv::Size& size);
    void letterbox(
        const cv::Mat& image,
        cv::Mat& out,
        cv::Size& size
    );
    void infer();
    void postprocess(std::vector<std::vector<Object>>& objs);
    static void draw_objects(
        const cv::Mat& image,
        cv::Mat& res,
        const std::vector<Object>& objs,
        const std::vector<std::string>& CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS
    );
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;

    std::vector<PreParam> pparams;  // 支持多batch
private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };
    int bs = 1;
    int dtype_size = 4;

};

YOLOv8::YOLOv8(const std::string& engine_file_path, int batch_size)
{
    bs = batch_size;
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    nvinfer1::DataType dtype = this->engine->getBindingDataType(0);  // 获取第一个binding的数据类型
    dtype_size = type_to_size(dtype);

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string name = this->engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims = this->engine->getProfileDimensions(
                       i,
                       0,
                       nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);

        } else {
            dims = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }

}

YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK_CUDA_11(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK_CUDA_11(cudaFreeHost(ptr));
    }

}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        /* for cuda >= 11.4*/
        CHECK_CUDA_11(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        //CHECK_CUDA_11(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));  // for cuda <= 11.3
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void* d_ptr, * h_ptr;
        size_t size = bindings.size * bindings.dsize;
        /* for cuda >= 11.4*/
        CHECK_CUDA_11(cudaMallocAsync(&d_ptr, size, this->stream));
        //CHECK_CUDA_11(cudaMalloc(&d_ptr, size));  // for cuda <= 11.3
        CHECK_CUDA_11(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size = bindings.size * bindings.dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK_CUDA_11(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size,
                                              cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");

    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    } else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    PreParam pparam;
    pparam.ratio = 1 / r;
    pparam.dw = dw;
    pparam.dh = dh;
    pparam.height = height;
    pparam.width = width;;
    this->pparams.push_back(pparam);  // 收集batch内多张图片的缩放信息
}


void YOLOv8::copy_from_Mat(const std::vector<cv::Mat>& images, cv::Size& size)
{
    cv::Mat nchw;
    this->context->setBindingDimensions(0, nvinfer1::Dims {4, { bs, 3, size.height, size.width } });

    int total_buffer_size = bs * 3 * size.height * size.width * dtype_size;
    float* temp_imgs = (float*)malloc(bs * 3 * size.height * size.width * dtype_size);

    for (int img_id = 0; img_id < bs; img_id++) {
        this->letterbox(images[img_id], nchw, size);
        int total_offset = nchw.total() * img_id;  // 指针偏移量不需要乘以数据类型的大小
        memcpy(temp_imgs + total_offset, nchw.ptr<float>(), nchw.total() * nchw.elemSize());  // 内存拷贝的时候需要乘以数据类型的大小

    }
    CHECK_CUDA_11(cudaMemcpyAsync(
                      this->device_ptrs[0],  // 目标地址
                      temp_imgs,
                      total_buffer_size,
                      cudaMemcpyHostToDevice,
                      this->stream)
                 );

}

void YOLOv8::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK_CUDA_11(cudaMemcpyAsync(this->host_ptrs[i],
                                      this->device_ptrs[i + this->num_inputs],
                                      osize,
                                      cudaMemcpyDeviceToHost,
                                      this->stream));

    }
    cudaStreamSynchronize(this->stream);

}

void YOLOv8::postprocess(std::vector<std::vector<Object>>& batch_objs)
{
    batch_objs.clear();
    int* batch_num_dets = static_cast<int*>(this->host_ptrs[0]);
    auto* batch_boxes = static_cast<float*>(this->host_ptrs[1]);
    auto* batch_scores = static_cast<float*>(this->host_ptrs[2]);
    int* batch_labels = static_cast<int*>(this->host_ptrs[3]);

    PreParam pparam;
    for (int img_id = 0; img_id < bs; img_id++) {
        int num_dets = batch_num_dets[img_id];
        auto* boxes = &batch_boxes[img_id * 4 * 100];  // 每张图片最多预测100个目标，每个目标4个坐标参数
        auto* scores = &batch_scores[img_id * 100];  // 每张图片最多预测100个目标，每个目标1个得分
        int* labels = &batch_labels[img_id * 100]; // 每张图片最多预测100个目标，每个目标1个预测类别
        pparam = this->pparams[img_id];
        auto& dw = pparam.dw;
        auto& dh = pparam.dh;
        auto& width = pparam.width;
        auto& height = pparam.height;
        auto& ratio = pparam.ratio;
        std::vector<Object> objs;
        for (int i = 0; i < num_dets; i++) {
            float* ptr = boxes + i * 4;

            float x0 = *ptr++ - dw;
            float y0 = *ptr++ - dh;
            float x1 = *ptr++ - dw;
            float y1 = *ptr - dh;

            x0 = clamp(x0 * ratio, 0.f, width);
            y0 = clamp(y0 * ratio, 0.f, height);
            x1 = clamp(x1 * ratio, 0.f, width);
            y1 = clamp(y1 * ratio, 0.f, height);
            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.prob = *(scores + i);
            obj.label = *(labels + i);
            objs.push_back(obj);
        }
        batch_objs.push_back(objs);
    }

}

void YOLOv8::draw_objects(
    const cv::Mat& image,
    cv::Mat& res,
    const std::vector<Object>& objs,
    const std::vector<std::string>& CLASS_NAMES,
    const std::vector<std::vector<unsigned int>>& COLORS
)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), { 0, 0, 255 }, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);
    }
}
