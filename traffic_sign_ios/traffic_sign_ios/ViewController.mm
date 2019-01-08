//
//  ViewController.mm
//  squeenzecnn
//
//  Created by dang on 2017/7/28.
//  Copyright © 2017年 dang. All rights reserved.
//

#import "ViewController.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#include <algorithm>
#include <functional>
#include <vector>
#include "net.h"

@interface ViewController ()
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
}

NSString *print_topk(const std::vector<float>& cls_scores, int topk, std::vector<std::string> labels)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());
    
    // print topk and score
    NSString *result = @"";
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        //        fprintf(stderr, "%d = %f\n", index, score);
        std::string label = labels[index];
        NSString *string = [NSString stringWithUTF8String:label.c_str()];
        result = [result stringByAppendingFormat:@"%@ %f \n", string, score];
        NSLog(@"label: %@ prob: %f", string, score);
    }
    return result;
}

int load_labels(std::string path, std::vector<std::string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");
    
    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);
        std::string str_s(str);
        
        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}

- (IBAction)predict:(id)sender {
    double total_latency = 0;
    int total_count = 0;
    // load image
    UIImage* image = [UIImage imageNamed:@"test.jpg"];
    // get rgba pixels from image
    int w = image.size.width;
    int h = image.size.height;
    fprintf(stderr, "%d x %d\n", w, h);
    unsigned char* rgba = new unsigned char[w*h*4];
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w*4,
                                                        colorSpace,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
        
        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }
////
    NSString *paramPath = [[NSBundle mainBundle] pathForResource:@"squeezenet_v1.1" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"squeezenet_v1.1" ofType:@"bin"];
//    NSString *paramPath = [[NSBundle mainBundle] pathForResource:@"traffic_sign" ofType:@"param"];
//    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"traffic_sign" ofType:@"bin"];
//    
    // init net
    double start = [[NSDate new] timeIntervalSince1970];
    ncnn::Net net;
    {
        int r0 = net.load_param([paramPath UTF8String]);
        int r1 = net.load_model([binPath UTF8String]);
        fprintf(stderr, "net load %d %d\n", r0, r1);
    }
    double end = [[NSDate new] timeIntervalSince1970];
    total_latency += (end - start);
    total_count += 1;
    NSLog(@"Time for init network: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
          total_count);
    std::vector<std::string> labels;
    NSString *labelPath = [[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"];
    load_labels([labelPath UTF8String], labels);
    
    // run forward
    start = [[NSDate new] timeIntervalSince1970];
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(4);
        
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2BGR, w, h, 112, 112);

        const float mean_vals[3] = {231.44722f, 122.618225f, 122.520485f}; //} {104.f, 117.f, 123.f};
        // in.substract_mean_normalize(mean_vals, 0);
        const float norms[3] = {1/128.f, 1/128.f, 1/128.f};
        in.substract_mean_normalize(mean_vals, norms);
        
        ex.input("data", in);
        ex.extract("prob", out);
//        ex.input("input_1_0", in);
//        ex.extract("activation_1_Softmax_01", out);
    }
    total_latency = 0;
    total_count = 0;
    end = [[NSDate new] timeIntervalSince1970];
    total_latency += (end - start);
    total_count += 1;
    NSLog(@"Time for forward network: %.4lf, avg: %.4lf, count: %d", end - start, total_latency / total_count,
          total_count);
    // get prob
    std::vector<float> cls_scores;
    {
        NSLog(@"out.c: %d", out.c);
        cls_scores.resize(out.c);
        for (int j=0; j<out.c; j++)
        {
            const float* prob = out.data + out.cstep * j;
            cls_scores[j] = prob[0];
            NSLog(@"%i %f", j, *prob);
        }
    }
    NSString *result = print_topk(cls_scores, 1, labels);
    self.resultLabel.text = result;
    // clean
    delete[] rgba;
}

@end
