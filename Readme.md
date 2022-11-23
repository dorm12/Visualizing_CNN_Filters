! git clone https://github.com/dorm12/Visualizing_CNN_Filters.git
from Visualizing_CNN_Filters.API import main as Visualizer

vslr = Visualizer.Visualizer(model, conv_layer_name='block3_conv3')
loss, img = vslr.feature_visualization(filter_index=3)

The above code is the suggested way to use the tool.
The 'img' will be the visualization of the given filter in the given layer in the given model.


methods available using Visualizer:
set_target_layer
create feature extractor
get_activations
evaluate_model
evaluate_model_without_filter
print_activations
gradcam_heatmap(self, data) - doesn't ready yey
cluster_activations
feature_visualization
get_feature_extractor
get_target_layer
get_model
