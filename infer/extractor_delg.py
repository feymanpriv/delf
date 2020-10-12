"""
/**
 * @file extractor.py
 * @author feymanpriv(547559398@qq.com)
 * @date 2020/9/21 11:16:30
 * @brief simple delg extractor based on tensorflow version
 **/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf


def MakeExtractor(cfg):
    """Creates a function to extract global and/or local features from an image.

    Args:
        config: DelfConfig proto containing the model configuration.

    Returns:
        Function that receives an image and returns features.

    Raises:
        ValueError: if config is invalid.
    """
    
    model = tf.saved_model.load(cfg.model_path)
    image_scales = list(map(float, cfg.image_scales.strip().split(",")))
    image_scales_tensor = tf.convert_to_tensor(image_scales)

    feeds = ['input_image:0', 'input_scales:0']
    fetches = []

    # Custom configuration needed when local features are used.
    # Extra input/output end-points/tensors.
    feeds.append('input_abs_thres:0')
    feeds.append('input_max_feature_num:0')
    fetches.append('boxes:0')
    fetches.append('features:0')
    fetches.append('scales:0')
    fetches.append('scores:0')
    score_threshold_tensor = tf.constant(float(cfg.score_threshold))
    max_feature_num_tensor = tf.constant(int(cfg.max_feature_num))

    if cfg.use_global_features:
        feeds.append('input_global_scales_ind:0')
        fetches.append('global_descriptors:0')

        global_scales_ind_tensor = tf.range(len(image_scales))
    

    def ExtractorFn(image_tensor):
        extracted_features = {}
        otuput = None

        predict = model.signatures['serving_default']
        if cfg.use_global_features:
            output_dict = predict(
                input_image=image_tensor,
                input_scales=image_scales_tensor,
                input_max_feature_num=max_feature_num_tensor,
                input_abs_thres=score_threshold_tensor,
                input_global_scales_ind=global_scales_ind_tensor)
            output = [
                output_dict['boxes'], output_dict['features'],
                output_dict['scales'], output_dict['scores'],
                output_dict['global_descriptors']
            ]
        
        if cfg.use_global_features:
            raw_global_descriptors = output[-1]
            global_descriptors = tf.nn.l2_normalize(
                raw_global_descriptors, axis=1, name='l2_normalization')
            global_descriptors = tf.reduce_sum(
                global_descriptors, axis=0, name='sum_pooling')
            final_global_descriptors = tf.nn.l2_normalize(
                global_descriptors, axis=0, name='final_l2_normalization')
            extracted_features.update({
                'global_descriptor': final_global_descriptors.numpy(),
            })
            
        boxes = output[0]
        raw_local_descriptors = output[1]
        feature_scales = output[2]
        attention_with_extra_dim = output[3]

        attention = tf.reshape(attention_with_extra_dim,
                                [tf.shape(attention_with_extra_dim)[0]])

        locations = CalculateKeypointCenters(boxes)
        local_descriptors = tf.nn.l2_normalize(
                raw_local_descriptors, axis=1, name='l2_normalization')

        extracted_features.update({
                'local_features': {
                'locations': locations.numpy(),
                'descriptors': local_descriptors.numpy(),
                'scales': feature_scales.numpy(),
                'attention': attention.numpy(),
                } 
        })

        return extracted_features
    
    return ExtractorFn


def CalculateKeypointCenters(boxes):
    """
    Args:
        boxes: [N, 4] float tensor.

    Returns:
        centers: [N, 2] float tensor.
    """
    return tf.divide(
             tf.add(
               tf.gather(boxes, [0, 1], axis=1), 
               tf.gather(boxes, [2, 3], axis=1)
             ), 2.0
           )
