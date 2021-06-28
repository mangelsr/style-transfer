import PIL.Image
import numpy as np
import tensorflow as tf


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# content_path = tf.keras.utils.get_file('selfie.jpg', './img/selfie.jpg')
content_path = './img/selfie.jpg'

# style_path = tf.keras.utils.get_file('mona lisa.jpg', './img/mona lisa.jpg')
style_path = './img/autoretrato.jpg'

content_image = load_img(content_path)
style_image = load_img(style_path)

transfer_model = tf.keras.models.load_model('./saved_model')
stylized_image = transfer_model(tf.constant(
    content_image), tf.constant(style_image))[0]

image_result = tensor_to_image(stylized_image)

image_result.save('result.png', 'png')
image_result.save('result.jpeg', 'jpeg')
