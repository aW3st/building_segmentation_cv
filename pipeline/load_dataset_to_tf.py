def normalize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (128, 128))
    input_mask = tf.image.resize(input_mask, (128, 128))
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32)
    print(input_image.shape, input_mask.shape)
    # input_mask -= 1
    return input_image, input_mask


def process_path(image_path):
    # trim 'i.jpg' from path and replace with 'mask.jpg'
    mask_path = tf.strings.regex_replace(image_path, '\1_(i).jpg', 'mask')
    
    # This will return a tuple of input & mask as the dataset format requires
    image_string = tf.io.read_file(image_path)
    mask_string = tf.io.read_file(mask_path)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    mask_decoded = tf.image.decode_jpeg(mask_string, channels=1)
    # print('image decoded type', type(image_decoded))


    # input_image, input_mask = tf.image.decode_jpeg(, tf.image.decode_jpeg(

    # image, mask = normalize(image_decoded, mask_decoded)

    # print(image.shape, mask.shape)


    return image_decoded, mask_decoded

# def data_gen(X=None, y=None, batch_size=32, nb_epochs=1, sess=None):
def generate_dataset_from_local(split='train'):
    
    # Create tensorflow dataset generator from directory of training examples:
    filepaths_ds = tf.data.Dataset.list_files('data/'+split+'/*[i]*')
    # print('sample file string tensor: ', next(iter(filepaths_ds)))
    labeled_ds = filepaths_ds.map(process_path)
    # print(type(labeled_ds))

    # # Test generator:
    # for f in train_ds.take(5):
    #     print(f.numpy())

    # Test proper read of image binaries
    # for image_raw, label_raw in dataset.take(1):
    #     print(repr(image_raw.numpy()[:100]))
    #     print()
    #     print(repr(label_raw.numpy()[:100]))

    return labeled_ds