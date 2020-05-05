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

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

# Mysql
import pymysql
import pymysql.cursors
from PIL import Image
import PIL.Image
from io import BytesIO
import cv2

db = pymysql.connect(user='root', password='control1234', host='localhost', database='nao_db', cursorclass=pymysql.cursors.DictCursor)
db.autocommit(True)
cur=db.cursor()


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "C:/Users/win8/Documents/PF/im2txt/model/train/model.ckpt-3000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "C:/Users/win8/Documents/PF/im2txt/data/mscoco/word_counts3.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

def getLastId():
  cur.execute("SELECT id FROM images ORDER BY id DESC LIMIT 1")
  data_id = cur.fetchall()
  if data_id:
    last_id = data_id[0]['id']
  else:
    last_id = 0
  return last_id


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
    
    old_id = getLastId()

    img_path = "C:/Users/win8/Documents/PF/image2text/imagenes/"

    for index in range(1,7):
      path = img_path+str(index)+".jpg"
      img = cv2.imread(path)
      with tf.gfile.GFile(path, "r") as f:
        image = f.read()

      captions = generator.beam_search(sess, image)
      #print("Captions for image %s:" % os.path.basename(filename))
      caption = captions[0]
        # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      cv2.imshow('Image', img)
        #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
      cur.execute("INSERT INTO audiotext(audiotextcol) VALUES (%s)",(sentence))
      db.commit();
      cv2.waitKey(9500)
"""
    while True:
      try:
        new_id = getLastId()
        if new_id>old_id:
          print(new_id)
          old_id = new_id
          cur.execute("SELECT image FROM images ORDER BY id DESC LIMIT 1")
          data = cur.fetchall()
          file_like=BytesIO(data[0]['image']) #tf.gfile.Glob(file_pattern)
          image = file_like.getvalue()
          #img=PIL.Image.open(file_like)
          #image = tf.image.decode_jpeg(str(list(img.getdata())));
          
          captions = generator.beam_search(sess, image)
          #print("Captions for image %s:" % os.path.basename(filename))
          for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            cur.execute("INSERT INTO img2txt(id_image,caption,caption_p) VALUES (%s,%s,%s)",(old_id,sentence,math.exp(caption.logprob)))
            db.commit();
      except KeyboardInterrupt:
        db.close();
        break
"""


if __name__ == "__main__":
  tf.app.run()