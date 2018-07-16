import tensorflow as tf


def input_fn(file_path, augment=False):
    def _read_csv(line):
        image_path, target = None



