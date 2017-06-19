#include "detector_wrapper.h"

#include <iostream>

#include "cuda.h"
#include "network.h"
#include "parser.h"
#include "image.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "utils.h"

#define DOABS 1

namespace darknet
{

namespace
{

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    if(DOABS){
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}

void get_region_boxes(layer l, float *predictions, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i,j,n;

    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = darknet::get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

}

class Detector::Private
{
public:
    Private(const std::string &config, const std::string &weights, int n_batch, int gpu_id)
    {
#ifdef GPU
        cuda_set_device(gpu_id);
#endif

        _net = parse_network_cfg_batch((char*)(config.c_str()), n_batch);

        _error = error_code();

        if(!_error)
        {
            if(!weights.empty())
            {
                ::load_weights(&_net, (char*)(weights.c_str()));
            }

            _detection_layer = _net.layers[_net.n-1];

            int num = _detection_layer.w * _detection_layer.h * _detection_layer.n;

            _boxes = (box *)calloc(num, sizeof(box));
            _probs = (float**)calloc(num, sizeof(float *));

            for(int j = 0; j < num; ++j)
            {
                _probs[j] = (float*)calloc(_detection_layer.classes, sizeof(float *));
            }

            load_thread = 0;

            memset(&train_args, 0, sizeof(train_args));

            train_args.w = _net.w;
            train_args.h = _net.h;
            train_args.paths = 0;
            train_args.n = _net.batch * _net.subdivisions;
            train_args.m = 0;
            train_args.classes = _detection_layer.classes;
            train_args.jitter = _detection_layer.jitter;
            train_args.num_boxes = _detection_layer.max_boxes;
            train_args.d = &train_buffer;
            train_args.type = DETECTION_DATA;
            train_args.threads = 8;

            train_args.angle = _net.angle;
            train_args.exposure = _net.exposure;
            train_args.saturation = _net.saturation;
            train_args.hue = _net.hue;
        }
    }

    std::vector< std::vector<darknet::target_t> > detect_batch(unsigned char **pdata, int n_data, int w, int h, int c, int step,
                                                               float nms, float threshold) const
    {
        srand(2222222);

        std::vector< std::vector<darknet::target_t> > result(n_data);

        ::data batch_data;

        memset(&batch_data, 0, sizeof(::data));

        batch_data.shallow = 0;
        batch_data.X.rows = n_data;
        batch_data.X.vals = (float **)(calloc(batch_data.X.rows, sizeof(float*)));
        batch_data.X.cols = h*w*c;

        //

        for(int l = 0; l < n_data; ++l)
        {
            unsigned char *data = pdata[l];

            ::image image = make_image(w, h, c);

            int i, j, k, count=0;

            for(k = 0; k < c; ++k)
            {
                for(i = 0; i < h; ++i){
                    for(j = 0; j < w; ++j){
                        image.data[count++] = data[i*step + j*c + k]/255.;
                    }
                }
            }

            if(c == 3)
            {
                rgbgr_image(image);
            }

            batch_data.X.vals[l] = image.data;
        }

        //

        matrix output = network_predict_data(_net, batch_data);

        int num = _detection_layer.w * _detection_layer.h * _detection_layer.n;

        for(int k = 0; k < n_data; ++k)
        {
            darknet::get_region_boxes(_detection_layer, output.vals[k], 1, 1, threshold, _probs, _boxes, 0, 0);

            do_nms_sort(_boxes, _probs, num, _detection_layer.classes, nms);

            for(int i = 0; i < num; ++i)
            {
                int class_id = max_index(_probs[i], _detection_layer.classes);

                bool is_clutter = (_detection_layer.classes != 1) && (class_id == 0);

                if(!is_clutter)
                {
                    float prob = _probs[i][class_id];

                    if(prob > threshold)
                    {
                        box b = _boxes[i];

                        int left  = (b.x-b.w/2.)*w;
                        int right = (b.x+b.w/2.)*w;
                        int top   = (b.y-b.h/2.)*h;
                        int bot   = (b.y+b.h/2.)*h;

                        if(left < 0) left = 0;
                        if(right > w-1) right = w-1;
                        if(top < 0) top = 0;
                        if(bot > h-1) bot = h-1;

                        darknet::target_t target;

                        target.left = left;
                        target.right = right;
                        target.top = top;
                        target.bottom = bot;

                        target.prediction.label = class_id;
                        target.prediction.probability = prob;

                        // TODO alternatives

                        result[k].push_back(target);
                    }
                }
            }
        }

        free_data(batch_data);

        return result;
    }

    ~Private()
    {
        // free_network(_net);

        cfree(_boxes);

        for(int j = 0; j < _detection_layer.side * _detection_layer.side * _detection_layer.n; ++j)
        {
            cfree(_probs[j]);
        }

        cfree(_probs);

        //

        if(load_thread)
        {
            pthread_join(load_thread, 0);

            free_data(train_buffer);
        }

        delete [] train_args.paths;
    }

    network _net;
    layer _detection_layer;

    box *_boxes;
    float **_probs;

    load_args train_args;
    std::vector<std::string> train_data;
    pthread_t load_thread;
    data train_buffer;

    int _error;
};

Detector::Detector(const std::string &config, const std::string &weights, int n_batch, int gpu_id):
    _p(new Private(config, weights, n_batch, gpu_id))
{
    this->error = _p->_error;
    this->width = _p->_net.w;
    this->height = _p->_net.h;
    this->channels = _p->_net.c;
    this->batch = _p->_net.batch;
    this->classes = _p->_detection_layer.classes;
}

Detector::~Detector()
{
    delete _p;
}

std::vector<std::vector<darknet::target_t> > Detector::detect(unsigned char **pdata, int n_data, int w, int h, int c,
                                                              int step, float nms, float threshold) const
{
    return _p->detect_batch(pdata, n_data, w, h, c, step, nms, threshold);
}

void Detector::set_train_data(const std::vector<std::string> &data_list)
{
    // Дожидаемся, пока загрузятся предыдущие данные и удаляем их

    if(_p->load_thread)
    {
        pthread_join(_p->load_thread, 0);

        free_data(_p->train_buffer);
    }

    // Устанавливаем новый список данных

    _p->train_args.m = data_list.size();

    delete [] _p->train_args.paths;

    _p->train_data = data_list;

    _p->train_args.paths = new char*[data_list.size()];

    for(size_t i = 0; i < _p->train_data.size(); ++i)
    {
        _p->train_args.paths[i] = (char*)(_p->train_data[i].c_str());
    }

    // Загружаем часть данных в потоках

    _p->load_thread = load_data(_p->train_args);
}

train_stat_t Detector::train()
{
    // Дожидаемся загрузки данных

    pthread_join(_p->load_thread, 0);

    // Запоминаем данные

    data current_data = _p->train_buffer;

    // Начинаем загружать новую порцию данных

    _p->load_thread = load_data(_p->train_args);

    // Обучаем

    train_stat_t stat;

    stat.loss = train_network(_p->_net, current_data);

    // Удаляем загруженные ранее данные

    free_data(current_data);

    return stat;
}

void Detector::save_weights(const std::string &path) const
{
    ::save_weights(_p->_net, (char*)(path.c_str()));
}

bool Detector::load_weights(const std::string &path)
{
    ::load_weights(&_p->_net, (char*)(path.c_str()));

    return true;
}

}

darknet::CnnDetector *create_detector(const char *config, const char *weights, int n_batch, int gpu_id)
{
    return new darknet::Detector(config, weights, n_batch, gpu_id);
}

void destroy_detector(darknet::CnnDetector *detector)
{
    delete detector;
}
