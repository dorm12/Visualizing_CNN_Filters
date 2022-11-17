import tensorflow as tf
import numpy as np

def compute_loss(input_image, filter_index, feature_extractor):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img = img + learning_rate * grads
    return loss, img


def initialize_image(img_width, img_height, custom):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    if custom:
        return img
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index, feature_extractor, 
                     img_width=180, img_height=180, 
                     custom=True, initializer=None, 
                     iterations=30, learning_rate = 10.0,
                     return_decomposed=True):
    if initializer:
        img = tf.Variable(initializer(img_width, img_height))
    else:
        img = initialize_image(img_width, img_height, custom)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    if return_decomposed:
        # Decode the resulting input image
        img = deprocess_image(img[0].numpy(), custom)
        return loss, img
    
    else:
        return loss, np.clip(img[0].numpy(), 0, 1)


def deprocess_image(img, custom):
    if custom:
        return np.clip(img[25:-25, 25:-25, :], 0, 1)
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # # Convert to RGB array
    # img *= 255
    # img = np.clip(img, 0, 255).astype("uint8")
    return img


    # loss, img = visualize_filter(filter_index)
    #
    # return loss, img