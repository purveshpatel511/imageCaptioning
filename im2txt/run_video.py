# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf
import cv2
import numpy as np

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/home/sorelys/im2txt/model/train/model.ckpt-3000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/home/sorelys/im2txt/data/mscoco/word_counts3.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  """ 
  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
  """
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    
    video = cv2.VideoCapture(0)
    while (True):
      try:
        #file_like=BytesIO(data[0]['image']) #tf.gfile.Glob(file_pattern)
        #image = file_like.getvalue()
        _, img = video.read()
        cv2.imwrite("imagenes/camara.jpg",img)
        #image = img.tobytes() 
        """ ERROR: Invalid JPEG data, size 921600
           [[Node: decode/DecodeJpeg = DecodeJpeg[acceptable_fraction=1, channels=3, dct_method="", 
           fancy_upscaling=true, ratio=1, try_recover_truncated=false, _device="/job:localhost/replica:0/task:0/cpu:0"](_recv_image_feed_0)]]
        """
        #temp=BytesIO(img) 
        #image = temp.getvalue()
        image = cv2.imread("imagenes/camara.jpg")
        #print(image)
        #img=PIL.Image.open(img)
        #image = tf.image.decode_jpeg(image);

        captions = generator.beam_search(sess, image)
        #print("Captions for image %s:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
          # Ignore begin and end words.
          sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
          sentence = " ".join(sentence)
          print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        
        cv2.imshow('Image', img)
        cv2.waitKey(15)
      except KeyboardInterrupt:
        break



if __name__ == "__main__":
  tf.app.run()