from flask import Flask
from flask import request
import json
import os
import tensorflow as tf

from src import model, sample, encoder

app = Flask(__name__)


def before_first_request():
    global enc, context, output, graph, saver, ckpt
    model_name = 'gpt_de_assis'
    temperature = 0.5
    top_k = 40
    length = 40

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))


app.before_first_request(before_first_request)


@app.route("/gpt_de_assis")
def gpt_de_assis():
    q = request.args.get('q')
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, ckpt)
        context_tokens = enc.encode(q)
        out = sess.run(output, feed_dict={
            context: [context_tokens]
        })
        text = enc.decode(out[0])
        return text


@app.route("/hai")
def hai():
    return "Haaai!s"
