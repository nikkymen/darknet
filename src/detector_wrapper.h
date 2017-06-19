#ifndef DARKNET_DETECTOR_WRAPPER_H
#define DARKNET_DETECTOR_WRAPPER_H

#include <string>
#include <vector>

namespace darknet
{

struct train_stat_t
{
    float loss;
};

struct prediction_t
{
    int label;
    float probability;
};

struct target_t
{
    int top;
    int left;
    int bottom;
    int right;

    prediction_t prediction;

    std::vector<prediction_t> alternatives;
};

class CnnDetector
{
public:
    virtual ~CnnDetector() {}

    virtual std::vector< std::vector<target_t> >
    detect(unsigned char **pdata, int n_data, int w, int h, int c, int step,
           float nms, float threshold) const = 0;

    virtual void save_weights(const std::string &path) const = 0;
    virtual bool load_weights(const std::string &path) = 0;

    virtual void set_train_data(const std::vector<std::string> &data_list) = 0;

    virtual train_stat_t train() = 0;

    int width;
    int height;
    int channels;
    int batch;
    int classes;
    int error;
};

class Detector : public CnnDetector
{
public:
    Detector(const std::string &config, const std::string &weights, int n_batch, int gpu_id);
    virtual ~Detector();

    std::vector< std::vector<darknet::target_t> >
    detect(unsigned char **pdata, int n_data, int w, int h, int c, int step,
           float nms, float threshold) const;

    void save_weights(const std::string &path) const;
    bool load_weights(const std::string &path);

    void set_train_data(const std::vector<std::string> &data_list);

    train_stat_t train();

public:
    class Private;

    Private *_p;
};

}

#ifdef __cplusplus
extern "C" {

darknet::CnnDetector * create_detector(const char *config, const char *weights, int n_batch, int gpu_id);

void destroy_detector(darknet::CnnDetector *detector);

}

#endif

#endif // DARKNET_DETECTOR_WRAPPER_H
