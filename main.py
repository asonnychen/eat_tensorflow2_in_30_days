# import tensorflow as tf
# def main():
#     print('GPU:', tf.config.list_physical_devices('GPU'))

# if __name__ == "__main__":
#     main()


# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
#         logical = tf.config.experimental.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical, {len(logical)} Logical GPUs")
#     except RuntimeError as e: print(e)

import tensorflow as tf
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())