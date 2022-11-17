if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Visualizing_CNN_Filters.API import Visualizing_Filters as VF

class Visualizer:
  MAX_CONV_LAYER = 4
  PATH = 'drive/some_model_using_convolutinal_layers'
  PATH = 'drive/MyDrive/ITC_Final_Project/saved_models/CNN4_imgcropped'
  def __init__(self, model, conv_layer_name=None):
    self.__model = model

    if model.name == "resnet50v2":
      self.__custom = False
    else:
      self.__custom = True
    
    # target conv layer will either: 
    # (1) be given as input, 
    # (2) be the last convolutinal layer but not deeper than MAX_CONV_LAYER
    if conv_layer_name:
      layer = model.get_layer(name=conv_layer_name)
    else:
      conv_layers = [layer for layer in self.__model.layers if 'conv' in layer.name]
      number_of_conv_layers = len(conv_layers)
      number_of_target_layer = min([self.MAX_CONV_LAYER, number_of_conv_layers-1])
      layer = conv_layers[number_of_target_layer]

    # saves the target layer
    self.__target_conv_layer = layer

    # creates the feature extractor
    self.__feature_extractor = keras.Model(inputs=model.inputs,
                                           outputs=layer.output)
    
    # save weights of target layer
    w, b = layer.get_weights()
    self.__target_layer_w = w
    self.__target_layer_b = b

    # save the target layer location
    for i, layer in enumerate(model.layers):
      if layer.name == conv_layer_name:
        self.__target_location = i
        break
    
    
  def create_feature_extractor(self, layer_name):
      layer = self.__model.get_layer(layer_name)
      return keras.Model(inputs=self.__model.inputs, outputs=layer.output)
    

  def set_target_layer(self, conv_layer_name):
      layer = self.__model.get_layer(name=conv_layer_name)
      self.__target_conv_layer = layer
      self.__feature_extractor = keras.Model(inputs=model.inputs,
                                           outputs=layer.output)
                                           
                                           
  def get_activations(self, data, layer_name=None):
      """returns the activations and the stadndart devaitations of them along the different images,
         note: low stds may point on unneccesary filters"""
      if layer_name:
          feature_extractor = self.create_feature_extractor(layer_name)
      else:
          feature_extractor = self.__feature_extractor
      acts = feature_extractor(data).numpy().mean(axis=(1,2))
      stds = acts.std(axis=0)
      return acts, stds, stds.argsort()
      
      
  def evaluate_model(self, data, labels):
      self.__model.evaluate(data, labels)
      
      
  def evaluate_model_without_filter(self, data, labels, layer_name, filter_index, restore_weights=True):
      def set_filter_weigths_to_zero(w,b,i):
          # creates w and b with zeros for the i'th filter
          # and leave the other filters intact
          # maybe worth breaking the list comprehension for readability purpose
          new_w = np.stack([np.zeros_like(w[:,:,:,i]) if j==i else w[:,:,:,j] for j in range(w.shape[-1])], -1)
          new_b = np.stack([np.zeros_like(b[j]) if j==i else b[j] for j in range(w.shape[-1])], -1)
          return new_w, new_b
      
      print('evaluation before:')
      _, original_acc = self.__model.evaluate(data, labels)
      
      w,b = self.__model.get_layer(layer_name).weights
      w, b = tf.identity(w), tf.identity(b)
    #   new_w[...,filter_index] = tf.zeros_like(new_w[...,filter_index])
    #   new_b[filter_index] = tf.zeros_like(new_b[filter_index])
      self.__model.get_layer(layer_name).set_weights(set_filter_weigths_to_zero(w,b,filter_index))
    #   self.__model.get_layer(layer_name).set_weights([new_w, new_b])
      print('new evaluation:')
      _, new_acc = self.__model.evaluate(data, labels)
      
      if type(restore_weights) in (int, float):
        if new_acc+restore_weights < original_acc:
          print(new_acc+restore_weights, original_acc)
          self.__model.get_layer(layer_name).set_weights([w,b])
      elif restore_weights==True:
          self.__model.get_layer(layer_name).set_weights([w,b])


  def print_activations(self, image, layer_name=None, layer_index=None, amount_of_filters=16):
    # :input: image, either layer's index or layer's name
    # prints activations of all filters in a given layer

    if layer_name:
      layer = self.__model.get_layer(name=layer_name)
    elif layer_index:
      layer = self.__model.layers[layer_index]
    else:
      print("a specific layer should be given, either by name or by index")

    activation_model = keras.Model(inputs=self.__model.input, outputs=layer.output)
    if len(image.shape)==4:
      activations = activation_model.predict(image)
    else:
      activations = activation_model.predict(image[np.newaxis, :])
    
    activation = activations[0]
    filter_index=0
    col_size = 4
    row_size = amount_of_filters // col_size + int(amount_of_filters>0)
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[:, :, filter_index], cmap='gray')
            filter_index += 1



  def gradcam_heatmap(self, data):
    # ...
    pass

  
  def cluster_activations(self, data=None, activations=None):
    # either data or activations should be given to the method
    # gives back best clustering of the inputs
    # uses sklearn whatever clustering method
    pass

  
  def feature_visualization(self, filter_index=None, amount_of_filters=None, 
                            img_width=180, img_height=180, initializer=None):
    # does feature visualiztion
    # uses another file
    s = VF.visualize_filter(filter_index, self.__feature_extractor, 
                            img_width, img_height, self.__custom, initializer=initializer)
    return s

  def redundancy_in_features(self, data, labels, var_1=None):
    # gets amount of redundant filters in the target conv layer
    # assumes fit has been done
    METRIC = 1
    original_evaluation = self.__model.evaluate(data, labels)
    self.__redundant = []

    def set_i_filter_weigths_to_zero(w,b,i):
      # creates w and b with zeros for the i'th filter
      # and leave the other filters intact
      # maybe worth breaking the list comprehension for readability purpose
      new_w = np.stack([np.zeros_like(w[:,:,:,i]) if j==i else w[:,:,:,j] for j in range(w.shape[-1])], -1)
      new_b = np.stack([np.zeros_like(b[j]) if j==i else b[j] for j in range(w.shape[-1])], -1)
      return new_w, new_b

    for i in self.__failed:
      model = keras.models.clone_model(self.__clone_model)
      new_w, new_b = set_i_filter_weigths_to_zero(self.__target_layer_w, self.__target_layer_b, i)
      model.layers[self.__target_location].set_weights([new_w, new_b])
      # print(f"{i}: ")
      new_eval = model.evaluate(data, labels)
      if new_eval[METRIC] >= original_evaluation[METRIC]:
        self.__redundant.append(i)

    return self.__redundant

    
  def get_feature_extractor(self):
    return self.__feature_extractor
  def get_conv_layer(self):
    return self.__target_conv_layer
  def get_model(self):
    return self.__model