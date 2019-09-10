#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import os
import json
import shutil
import logging
import datetime
import urllib.parse
import pandas as pd
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

from apache_beam.io import textio
from apache_beam.io import tfrecordio

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope
from tensorflow.python.lib.io import file_io
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders.csv_coder import CsvCoder
from tensorflow_transform.coders.example_proto_coder import ExampleProtoCoder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata.schema_utils import schema_from_feature_spec
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform import coders as tft_coders
from tensorflow_model_analysis.slicer import slicer

from ipywidgets.embed import embed_data

DATA_DIR = 'data/'


#  ------------------------------------------------------------------------------------------------
#                                    FUNCTIONS
#  ------------------------------------------------------------------------------------------------

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour',
    'trip_start_day',
    'trip_start_month'
]

DENSE_FLOAT_FEATURE_KEYS = [
    'trip_miles',
    'fare',
    'trip_seconds'
]

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'pickup_census_tract',
    'dropoff_census_tract',
    'payment_type',
    'company',
    'pickup_community_area',
    'dropoff_community_area'
]

# allow nan values in these features.
OPTIONAL_FEATURES = [
    'dropoff_latitude',
    'dropoff_longitude',
    'pickup_census_tract',
    'dropoff_census_tract',
    'company',
    'trip_seconds',
    'dropoff_community_area'
]

LABEL_KEY = 'tips'
FARE_KEY = 'fare'


def make_tft_input_metadata(schema):
    """Make a TFT Schema object
    Args:
      schema: schema list of training data.
    Returns:
      TFT metadata object.
    """
    tft_schema = {}
    for col_schema in schema:
        # Use VarLenFeature to allow for missing values (will be added in process function)
        if col_schema['type'] == 'NUMBER':
            if col_schema['name'] in OPTIONAL_FEATURES:
                tft_schema[col_schema['name']] = tf.io.VarLenFeature(tf.float32)
            else:
                tft_schema[col_schema['name']] = tf.io.FixedLenFeature([], tf.float32)
        elif col_schema['type'] in ['CATEGORY', 'TEXT', 'IMAGE_URL', 'KEY']:
            if col_schema['name'] in OPTIONAL_FEATURES:
                tft_schema[col_schema['name']] = tf.io.VarLenFeature(tf.string)
            else:
                tft_schema[col_schema['name']] = tf.io.FixedLenFeature([], tf.string)

    return dataset_metadata.DatasetMetadata(schema_from_feature_spec(tft_schema))


def to_dense(tensor):
    """Takes as input a SparseTensor and return a Tensor with correct default value
    Args:
      tensor: tf.SparseTensor
    Returns:
      tf.Tensor
    """
    if not isinstance(tensor, tf.sparse.SparseTensor):
        return tensor
    if tensor.dtype == tf.string:
        default_value = ''
    elif tensor.dtype == tf.float32:
        default_value = 0.0
    elif tensor.dtype == tf.int32:
        default_value = 0
    else:
        raise ValueError(f"Tensor type not recognized: {tensor.dtype}")

    return tf.squeeze(tf.sparse_to_dense(tensor.indices,
                               [tensor.dense_shape[0], 1],
                               tensor.values, default_value=default_value), axis=1)
    # TODO: Update to below version
    # return tf.squeeze(tf.sparse.to_dense(tensor, default_value=default_value), axis=1)


def preprocess(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        outputs[key] = tft.scale_to_z_score(to_dense(inputs[key]))

    for key in VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        if inputs[key].dtype == tf.string:
            vocab_tensor = to_dense(inputs[key])
        else:
            vocab_tensor = tf.as_string(to_dense(inputs[key]))
        outputs[key] = tft.compute_and_apply_vocabulary(
            vocab_tensor, vocab_filename='vocab_' + key,
            top_k=VOCAB_SIZE, num_oov_buckets=OOV_SIZE)

    for key in BUCKET_FEATURE_KEYS:
        outputs[key] = tft.bucketize(to_dense(inputs[key]), FEATURE_BUCKET_COUNT)

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tf.cast(to_dense(inputs[key]), tf.int64)

    taxi_fare = to_dense(inputs[FARE_KEY])
    taxi_tip = to_dense(inputs[LABEL_KEY])
    # Test if the tip was > 20% of the fare.
    tip_threshold = tf.multiply(taxi_fare, tf.constant(0.2))
    outputs[LABEL_KEY] = tf.logical_and(
        tf.logical_not(tf.math.is_nan(taxi_fare)),
        tf.greater(taxi_tip, tip_threshold))

    return outputs


def get_feature_columns():
    """Callback that returns a list of feature columns for building a tf.estimator.
    Returns:
      A list of tf.feature_column.
    """
    return (
            [tf.feature_column.numeric_column(key, shape=()) for key in DENSE_FLOAT_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    key, num_buckets=VOCAB_SIZE + OOV_SIZE))
                for key in VOCAB_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    key, num_buckets=FEATURE_BUCKET_COUNT, default_value=0))
                for key in BUCKET_FEATURE_KEYS] +
            [tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    key, num_buckets=num_buckets, default_value=0))
                for key, num_buckets in zip(CATEGORICAL_FEATURE_KEYS, MAX_CATEGORICAL_FEATURE_VALUES)])

#  ------------------------------------------------------------------------------------------------
#                                    DATA VALIDATION OP
#  ------------------------------------------------------------------------------------------------

# pass


#  ------------------------------------------------------------------------------------------------
#                                    TRANSFORM OP
#  ------------------------------------------------------------------------------------------------


def run_transform(output_dir, schema, train_data_file, eval_data_file, preprocessing_fn=None):
    """Writes a tft transform fn, and metadata files.
    Args:
      output_dir: output folder
      schema: schema list.
      train_data_file: training data file pattern.
      eval_data_file: eval data file pattern.
      preprocessing_fn: a function used to preprocess the raw data. If not
                        specified, a function will be automatically inferred
                        from the schema.
    """

    tft_input_metadata = make_tft_input_metadata(schema)
    temp_dir = os.path.join(output_dir, 'tmp')

    runner = 'DirectRunner'
    with beam.Pipeline(runner, options=None) as p:
        with beam_impl.Context(temp_dir=temp_dir):
            names = [x['name'] for x in schema]
            converter = CsvCoder(names, tft_input_metadata.schema)
            train_data = (
                    p
                    | 'ReadTrainData' >> textio.ReadFromText(train_data_file)
                    | 'DecodeTrainData' >> beam.Map(converter.decode))

            transformed_dataset, transform_fn = (
                    (train_data, tft_input_metadata) | beam_impl.AnalyzeAndTransformDataset(preprocessing_fn))
            transformed_data, transformed_metadata = transformed_dataset

            # Writes transformed_metadata and transfrom_fn folders
            # TODO: check out what is the transform function that came from previous step
            _ = (transform_fn | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(output_dir))

            # Write the raw_metadata
            metadata_io.write_metadata(
                metadata=tft_input_metadata,
                path=os.path.join(output_dir, 'metadata'))

            _ = transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
                os.path.join(output_dir, 'train'),
                coder=ExampleProtoCoder(transformed_metadata.schema))

            eval_data = (
                    p
                    | 'ReadEvalData' >> textio.ReadFromText(eval_data_file)
                    | 'DecodeEvalData' >> beam.Map(converter.decode))

            eval_dataset = (eval_data, tft_input_metadata)
            transformed_eval_dataset = (
                    (eval_dataset, transform_fn) | beam_impl.TransformDataset())
            transformed_eval_data, transformed_metadata = transformed_eval_dataset

            _ = transformed_eval_data | 'WriteEvalData' >> tfrecordio.WriteToTFRecord(
                os.path.join(output_dir, 'eval'),
                coder=ExampleProtoCoder(transformed_metadata.schema))


# TRAIN_DATA = 'taxi-cab-classification/train.csv'
# EVALUATION_DATA = 'taxi-cab-classification/eval.csv'
#
# if os.path.exists(os.path.join(DATA_DIR, "transformed")):
#     shutil.rmtree(os.path.join(DATA_DIR, "transformed"))
#
# logging.getLogger().setLevel(logging.INFO)
# schema = json.loads(file_io.read_file_to_string(os.path.join(DATA_DIR, "taxi-cab-classification/schema.json")))
#
#
# def wrapped_preprocessing_fn(inputs):
#     outputs = preprocess(inputs)
#     for key in outputs:
#         if outputs[key].dtype == tf.bool:
#             outputs[key] = tft.compute_and_apply_vocabulary(tf.as_string(outputs[key]),
#                                              vocab_filename='vocab_' + key)
#     return outputs
#
#
# preprocessing_fn = wrapped_preprocessing_fn
#
# run_transform(os.path.join(DATA_DIR, "transformed"),
#               schema,
#               DATA_DIR + TRAIN_DATA,
#               DATA_DIR + EVALUATION_DATA,
#               preprocessing_fn=preprocessing_fn)




#  ------------------------------------------------------------------------------------------------
#                                    TRAIN OP
#  ------------------------------------------------------------------------------------------------

LEARNING_RATE = 0.1
HIDDEN_LAYER_SIZE = '1500'
STEPS = 3
BATCH_SIZE = 32
EPOCHS = 1
CLASSIFICATION_TARGET_TYPES = [tf.bool, tf.int32, tf.int64]
REGRESSION_TARGET_TYPES = [tf.float32, tf.float64]
TARGET_TYPES = CLASSIFICATION_TARGET_TYPES + REGRESSION_TARGET_TYPES


def is_classification(transformed_output, target):
    """Whether the scenario is classification (vs regression).

    Returns:
      The number of classes if the target represents a classification
      problem, or None if it does not.
    """
    if target not in transformed_output.transformed_feature_spec():
        raise ValueError('Cannot find target "%s" in transformed data.' % target)

    feature = transformed_output.transformed_feature_spec()[target]
    if (not isinstance(feature, tf.io.FixedLenFeature) or feature.shape != [] or
            feature.dtype not in TARGET_TYPES):
        raise ValueError('target "%s" is of invalid type.' % target)

    if feature.dtype in CLASSIFICATION_TARGET_TYPES:
        if feature.dtype == tf.bool:
            return 2
        return transformed_output.vocabulary_size_by_name("vocab_" + target)

    return None


def make_training_input_fn(transformed_output, transformed_examples, batch_size, target_name):
    """Creates an input function reading from transformed data.
    Args:
      transformed_output: tft.TFTransformOutput
      transformed_examples: Base filename of examples
      batch_size: Batch size.
      target_name: name of the target column.
    Returns:
      The input function for training or eval.
    """
    def _input_fn():
        """Input function for training and eval."""
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=transformed_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True)

        transformed_features = dataset.make_one_shot_iterator().get_next()

        # Extract features and label from the transformed tensors.
        transformed_labels = transformed_features.pop(target_name)

        return transformed_features, transformed_labels

    return _input_fn


def make_serving_input_fn(transformed_output):
    """Creates an input function reading from transformed data.
    Args:
      transformed_output: tft.TFTransformOutput
    Returns:
      The input function for serving.
    """
    raw_feature_spec = transformed_output.raw_feature_spec()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(LABEL_KEY)

    def _serving_input_fn():
        """Input function for serving."""
        # Get raw features by generating the basic serving input_fn and calling it.
        # Here we generate an input_fn that expects a parsed Example proto to be fed
        # to the model at serving time.  See also
        # tf.estimator.export.build_raw_serving_input_receiver_fn.
        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()

        # Apply the transform function that was used to generate the materialized data.
        raw_features = serving_input_receiver.features
        transformed_features = tf_transform_output.transform_raw_features(
            raw_features)

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, serving_input_receiver.receiver_tensors)

    return _serving_input_fn


def get_estimator(transformed_output, target_name, output_dir, hidden_units,
                  optimizer, learning_rate, feature_columns):
    """Get proper tf.estimator (DNNClassifier or DNNRegressor)."""
    if optimizer == 'Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    elif optimizer == 'SGD':
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'Adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate)
    else:
        raise ValueError(f"Optimizer value not recognized: {optimizer}")

    # Set how often to run checkpointing in terms of steps.
    config = tf.estimator.RunConfig(save_checkpoints_steps=1000)
    n_classes = is_classification(transformed_output, target_name)
    if n_classes:
        estimator = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=n_classes,
            config=config,
            model_dir=output_dir)
    else:
        estimator = tf.estimator.DNNRegressor(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            config=config,
            model_dir=output_dir,
            optimizer=optimizer)

    return estimator


def eval_input_receiver_fn(transformed_output, target):
    """Build everything needed for the tf-model-analysis to run the model.
    Args:
      transformed_output: tft.TFTransformOutput
      target: name of the target column.
    Returns:
      EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untranformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    serialized_tf_example = tf.compat.v1.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')
    features = tf.io.parse_example(serialized_tf_example, transformed_output.transformed_feature_spec())
    transformed_features = transformed_output.transform_raw_features(features)
    receiver_tensors = {'examples': serialized_tf_example}
    return tfma.export.EvalInputReceiver(
        features=transformed_features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[target])


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

hidden_layer_size = [int(x.strip()) for x in HIDDEN_LAYER_SIZE.split(',')]

transformed_data_dir = os.path.join(DATA_DIR, "transformed")
tf_transform_output = tft.TFTransformOutput(transformed_data_dir)

feature_columns = get_feature_columns()
estimator = get_estimator(tf_transform_output,
                          "tips",
                          os.path.join(DATA_DIR, "training"),
                          hidden_layer_size,
                          "Adagrad",
                          LEARNING_RATE,
                          feature_columns)

train_input_fn = make_training_input_fn(tf_transform_output, os.path.join(transformed_data_dir, 'train' + '*'),
                                        BATCH_SIZE, "tips")
eval_input_fn = make_training_input_fn(tf_transform_output, os.path.join(transformed_data_dir, 'eval' + '*'),
                                       BATCH_SIZE, "tips")
serving_input_fn = make_serving_input_fn(tf_transform_output)

exporter = tf.estimator.FinalExporter('export', serving_input_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=STEPS)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=[exporter])
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

eval_model_dir = os.path.join(DATA_DIR + "training", 'tfma_eval_model_dir')
tfma.export.export_eval_savedmodel(
    estimator=estimator,
    export_dir_base=eval_model_dir,
    eval_input_receiver_fn=(
        lambda: eval_input_receiver_fn(tf_transform_output, "tips")))

metadata = {
    'outputs': [{
        'type': 'tensorboard',
        'source': os.path.join(DATA_DIR, "training"),
    }]
}

try:
    os.makedirs(os.path.join(DATA_DIR, "artifacts", "training"))
except OSError:
    pass
with open(os.path.join(DATA_DIR, "artifacts", "training", 'mlpipeline-ui-metadata.json'), 'w') as f:
    json.dump(metadata, f)

#  ------------------------------------------------------------------------------------------------
#                                    ANALYZE OP
#  ------------------------------------------------------------------------------------------------
#
#
# _OUTPUT_HTML_FILE = 'output_display.html'
# _STATIC_HTML_TEMPLATE = """
# <html>
#   <head>
#     <title>TFMA Slicing Metrics</title>
#
#     <!-- Load RequireJS, used by the IPywidgets for dependency management -->
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
#             integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA="
#             crossorigin="anonymous">
#     </script>
#
#     <!-- Load IPywidgets bundle for embedding. -->
#     <script src="https://unpkg.com/@jupyter-widgets/html-manager@^0.12.0/dist/embed-amd.js"
#             crossorigin="anonymous">
#     </script>
#
#     <!-- Load IPywidgets bundle for embedding. -->
#     <script>
#       require.config({{
#         paths: {{
#           "tfma_widget_js": "https://cdn.rawgit.com/tensorflow/model-analysis/v0.6.0/tensorflow_model_analysis/static/index"
#         }}
#       }});
#     </script>
#
#     <link rel="import" href="https://cdn.rawgit.com/tensorflow/model-analysis/v0.6.0/tensorflow_model_analysis/static/vulcanized_template.html">
#
#     <!-- The state of all the widget models on the page -->
#     <script type="application/vnd.jupyter.widget-state+json">
#       {manager_state}
#     </script>
#   </head>
#
#   <body>
#     <h1>TFMA Slicing Metrics</h1>
#     {widget_views}
#   </body>
# </html>
# """
# _SINGLE_WIDGET_TEMPLATE = """
#     <div id="slicing-metrics-widget-{0}">
#       <script type="application/vnd.jupyter.widget-view+json">
#         {1}
#       </script>
#     </div>
# """
#
#
# def get_raw_feature_spec(schema):
#     feature_spec = {}
#     for column in schema:
#         column_name = column['name']
#         column_type = column['type']
#
#         feature = tf.FixedLenFeature(shape=[], dtype=tf.string, default_value='')
#         if column_type == 'NUMBER':
#             feature = tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0)
#         feature_spec[column_name] = feature
#     return feature_spec
#
#
# def clean_raw_data_dict(raw_feature_spec):
#     def clean_method(input_dict):
#         output_dict = {}
#
#         for key in raw_feature_spec:
#             if key not in input_dict or not input_dict[key]:
#                 output_dict[key] = raw_feature_spec[key].default_value
#             else:
#                 output_dict[key] = input_dict[key]
#         return output_dict
#
#     return clean_method
#
#
# def run_analysis(output_dir, model_dir, eval_path, schema, slice_columns):
#     runner = 'DirectRunner'
#     column_names = [x['name'] for x in schema]
#     for slice_column in slice_columns:
#         if slice_column not in column_names:
#             raise ValueError("Unknown slice column: %s" % slice_column)
#
#     slice_spec = [
#         slicer.SingleSliceSpec(),  # An empty spec is required for the 'Overall' slice
#         slicer.SingleSliceSpec(columns=slice_columns)
#     ]
#
#     with beam.Pipeline(runner=runner, options=None) as pipeline:
#         raw_feature_spec = get_raw_feature_spec(schema)
#         raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
#         example_coder = tft_coders.example_proto_coder.ExampleProtoCoder(raw_schema)
#         csv_coder = tft_coders.CsvCoder(column_names, raw_schema)
#
#         raw_data = (
#                 pipeline
#                 | 'ReadFromText' >> beam.io.ReadFromText(eval_path)
#                 | 'ParseCSV' >> beam.Map(csv_coder.decode)
#                 | 'CleanData' >> beam.Map(clean_raw_data_dict(raw_feature_spec))
#                 | 'ToSerializedTFExample' >> beam.Map(example_coder.encode)
#                 | 'EvaluateAndWriteResults' >> tfma.EvaluateAndWriteResults(
#             eval_saved_model_path=model_dir,
#             slice_spec=slice_spec,
#             output_path=output_dir))
#
#
# def generate_static_html_output(output_dir, slicing_columns):
#     result = tfma.load_eval_result(output_path=output_dir)
#     slicing_metrics_views = [
#         tfma.view.render_slicing_metrics(result, slicing_column=slicing_column)
#         for slicing_column in slicing_columns
#     ]
#     data = embed_data(views=slicing_metrics_views)
#     manager_state = json.dumps(data['manager_state'])
#     widget_views = [json.dumps(view) for view in data['view_specs']]
#     views_html = ""
#     for idx, view in enumerate(widget_views):
#         views_html += _SINGLE_WIDGET_TEMPLATE.format(idx, view)
#     rendered_template = _STATIC_HTML_TEMPLATE.format(
#         manager_state=manager_state, widget_views=views_html)
#     static_html_path = os.path.join(output_dir, _OUTPUT_HTML_FILE)
#     file_io.write_string_to_file(static_html_path, rendered_template)
#
#     metadata = {
#         'outputs': [{
#             'type': 'web-app',
#             'storage': 'gcs',
#             'source': static_html_path,
#         }]
#     }
#     try:
#         os.makedirs(os.path.join(DATA_DIR, "artifacts", "analyze"))
#     except OSError:
#         pass
#     with file_io.FileIO(os.path.join(DATA_DIR, 'artifacts', 'analyze', 'mlpipeline-ui-metadata.json'), 'w') as f:
#         json.dump(metadata, f)
#
#
# # In[ ]:
#
#
# SLICE_COLUMN = ['trip_start_hour']
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
# slice_columns = [
#     column
#     for column_group in SLICE_COLUMN
#     for column in column_group.split(',')
# ]
# schema = json.loads(file_io.read_file_to_string(os.path.join(DATA_DIR, "taxi-cab-classification/schema.json")))
# eval_model_parent_dir = os.path.join(DATA_DIR, "training", 'tfma_eval_model_dir')
# model_export_dir = os.path.join(eval_model_parent_dir, file_io.list_directory(eval_model_parent_dir)[0])
# run_analysis(os.path.join(DATA_DIR, "analysis"),
#              model_export_dir,
#              DATA_DIR + EVALUATION_DATA,
#              schema,
#              slice_columns)
# generate_static_html_output(os.path.join(DATA_DIR, "analysis"), slice_columns)
#
#
# # ## 5. Predict Op
#
# # In[ ]:
#
#
# class EmitAsBatchDoFn(beam.DoFn):
#     """A DoFn that buffers the records and emits them batch by batch."""
#
#     def __init__(self, batch_size):
#         self._batch_size = batch_size
#         self._cached = []
#
#     def process(self, element):
#         from apache_beam.transforms import window
#         from apache_beam.utils.windowed_value import WindowedValue
#         self._cached.append(element)
#         if len(self._cached) >= self._batch_size:
#             emit = self._cached
#             self._cached = []
#             yield emit
#
#     def finish_bundle(self, context=None):
#         from apache_beam.transforms import window
#         from apache_beam.utils.windowed_value import WindowedValue
#         if len(self._cached) > 0:
#             yield WindowedValue(self._cached, -1, [window.GlobalWindow()])
#
#
# class TargetToLastDoFn(beam.DoFn):
#     """A DoFn that moves specified target column to last."""
#
#     def __init__(self, names, target_name):
#         self._names = names
#         self._target_name = target_name
#         self._names_no_target = list(names)
#         self._names_no_target.remove(target_name)
#
#     def process(self, element):
#         import csv
#         content = next(csv.DictReader([element], fieldnames=self._names))
#         target = content.pop(self._target_name)
#         yield [content[x] for x in self._names_no_target] + [target]
#
#
# class PredictDoFn(beam.DoFn):
#     """A DoFn that performs predictions with given trained model."""
#
#     def __init__(self, model_export_dir):
#         self._model_export_dir = model_export_dir
#
#     def start_bundle(self):
#         from tensorflow.contrib import predictor
#
#         # We need to import the tensorflow_transform library in order to
#         # register all of the ops that might be used by a saved model that
#         # incorporates TFT transformations.
#         import tensorflow_transform
#
#         self._predict_fn = predictor.from_saved_model(self._model_export_dir)
#
#     def process(self, element):
#         import csv
#         import io
#
#         prediction_inputs = []
#         for instance in element:
#             instance_copy = list(instance)
#             instance_copy.pop()  # remove target
#             buf = io.StringIO()
#             writer = csv.writer(buf, lineterminator='')
#             writer.writerow(instance_copy)
#             prediction_inputs.append(buf.getvalue())
#
#         return_dict = self._predict_fn({"inputs": prediction_inputs})
#         return_dict['source'] = element
#         yield return_dict
#
#
# class ListToCsvDoFn(beam.DoFn):
#     """A DoFn function that convert list to csv line."""
#
#     def process(self, element):
#         import csv
#         import io
#         buf = io.StringIO()
#         writer = csv.writer(buf, lineterminator='')
#         writer.writerow(element)
#         yield buf.getvalue()
#
#
# # In[ ]:
#
#
# def run_predict(output_dir, data_path, schema, target_name, model_export_dir, batch_size):
#     """Run predictions with given model using DataFlow.
#     Args:
#       output_dir: output folder
#       data_path: test data file path.
#       schema: schema list.
#       target_name: target column name.
#       model_export_dir: GCS or local path of exported model trained with tft preprocessed data.
#       batch_size: batch size when running prediction.
#     """
#
#     target_type = next(x for x in schema if x['name'] == target_name)['type']
#     labels_file = os.path.join(model_export_dir, 'assets', 'vocab_' + target_name)
#     is_classification = file_io.file_exists(labels_file)
#
#     output_file_prefix = os.path.join(output_dir, 'prediction_results')
#     output_schema_file = os.path.join(output_dir, 'schema.json')
#     names = [x['name'] for x in schema]
#
#     output_schema = [x for x in schema if x['name'] != target_name]
#     if is_classification:
#         with file_io.FileIO(labels_file, mode='r') as f:
#             labels = [x.strip() for x in f.readlines()]
#
#         output_schema.append({'name': 'target', 'type': 'CATEGORY'})
#         output_schema.append({'name': 'predicted', 'type': 'CATEGORY'})
#         output_schema.extend([{'name': x, 'type': 'NUMBER'} for x in labels])
#     else:
#         output_schema.append({'name': 'target', 'type': 'NUMBER'})
#         output_schema.append({'name': 'predicted', 'type': 'NUMBER'})
#
#     runner = 'DirectRunner'
#     with beam.Pipeline(runner, options=None) as p:
#         raw_results = (p
#                        | 'read data' >> beam.io.ReadFromText(data_path)
#                        | 'move target to last' >> beam.ParDo(TargetToLastDoFn(names, target_name))
#                        | 'batch' >> beam.ParDo(EmitAsBatchDoFn(batch_size))
#                        | 'predict' >> beam.ParDo(PredictDoFn(model_export_dir)))
#
#         if is_classification:
#             processed_results = (raw_results
#                                  | 'unbatch' >> beam.FlatMap(lambda x: list(zip(x['source'], x['scores'])))
#                                  | 'get predicted' >> beam.Map(lambda x: x[0] + [labels[x[1].argmax()]] + list(x[1])))
#         else:
#             processed_results = (raw_results
#                                  | 'unbatch' >> beam.FlatMap(lambda x: list(zip(x['source'], x['outputs'])))
#                                  | 'get predicted' >> beam.Map(lambda x: x[0] + list(x[1])))
#
#         results_save = (processed_results
#                         | 'write csv lines' >> beam.ParDo(ListToCsvDoFn())
#                         | 'write file' >> beam.io.WriteToText(output_file_prefix))
#
#         (results_save
#          | 'fixed one' >> beam.transforms.combiners.Sample.FixedSizeGlobally(1)
#          | 'set schema' >> beam.Map(lambda path: json.dumps(output_schema))
#          | 'write schema file' >> beam.io.WriteToText(output_schema_file, shard_name_template=''))
#
#
# logging.getLogger().setLevel(logging.INFO)
#
# # Models trained with estimator are exported to base/export/export/123456781 directory.
# # Our trainer export only one model.
# model = os.path.join(DATA_DIR, 'training')  # args.model
# export_parent_dir = os.path.join(model, 'export', 'export')
# model_export_dir = os.path.join(export_parent_dir, file_io.list_directory(export_parent_dir)[0])
# schema = json.loads(file_io.read_file_to_string(os.path.join(DATA_DIR, "taxi-cab-classification/schema.json")))
#
# batchsize = 32
# run_predict(os.path.join(DATA_DIR, "predict"),
#             os.path.join(DATA_DIR, EVALUATION_DATA),
#             schema,
#             "tips",
#             model_export_dir,
#             batchsize)
# prediction_results = os.path.join(DATA_DIR + "predict", 'prediction_results-*')
#
# with file_io.FileIO(os.path.join(DATA_DIR, "predict", 'schema.json'), 'r') as f:
#     schema = json.load(f)
#
# metadata = {
#     'outputs': [{
#         'type': 'table',
#         'storage': 'gcs',
#         'format': 'csv',
#         'header': [x['name'] for x in schema],
#         'source': prediction_results
#     }]
# }
# try:
#     os.makedirs(os.path.join(DATA_DIR, "artifacts", "predict"))
# except OSError:
#     pass
# with open(os.path.join(DATA_DIR, 'artifacts', 'predict', 'mlpipeline-ui-metadata.json'), 'w') as f:
#     json.dump(metadata, f)
#
# # ## 6. Confusion Matrix Op
#
# # In[ ]:
#
#
# predictions = os.path.join(DATA_DIR, 'predict', 'prediction_results-*')
# target_lambda = """lambda x: (x['target'] > x['fare'] * 0.2)"""
# if not os.path.exists(os.path.join(DATA_DIR, "confusionmatrix")):
#     os.makedirs(os.path.join(DATA_DIR, "confusionmatrix"))
#
# schema_file = os.path.join(os.path.dirname(predictions), 'schema.json')
# schema = json.loads(file_io.read_file_to_string(schema_file))
# names = [x['name'] for x in schema]
# dfs = []
# files = file_io.get_matching_files(predictions)
# for file in files:
#     with file_io.FileIO(file, 'r') as f:
#         dfs.append(pd.read_csv(f, names=names))
#
# df = pd.concat(dfs)
# if target_lambda:
#     df['target'] = df.apply(eval(target_lambda), axis=1)
#
# vocab = list(df['target'].unique())
# cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
# data = []
# for target_index, target_row in enumerate(cm):
#     for predicted_index, count in enumerate(target_row):
#         data.append((vocab[target_index], vocab[predicted_index], count))
#
# df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
# cm_file = os.path.join(DATA_DIR, 'confusionmatrix', 'confusion_matrix.csv')
# with file_io.FileIO(cm_file, 'w') as f:
#     df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)
#
# metadata = {
#     'outputs': [{
#         'type': 'confusion_matrix',
#         'format': 'csv',
#         'schema': [
#             {'name': 'target', 'type': 'CATEGORY'},
#             {'name': 'predicted', 'type': 'CATEGORY'},
#             {'name': 'count', 'type': 'NUMBER'},
#         ],
#         'source': cm_file,
#         # Convert vocab to string because for bealean values we want "True|False" to match csv data.
#         'labels': list(map(str, vocab)),
#     }]
# }
# try:
#     os.makedirs(os.path.join(DATA_DIR, "artifacts", "confusionmatrix"))
# except OSError:
#     pass
# with file_io.FileIO(os.path.join(DATA_DIR, 'artifacts', 'confusionmatrix', 'mlpipeline-ui-metadata.json'), 'w') as f:
#     json.dump(metadata, f)
#
# accuracy = accuracy_score(df['target'], df['predicted'])
# metrics = {
#     'metrics': [{
#         'name': 'accuracy-score',
#         'numberValue': accuracy,
#         'format': "PERCENTAGE",
#     }]
# }
# with file_io.FileIO(os.path.join(DATA_DIR, 'artifacts', 'confusionmatrix', 'mlpipeline-metrics.json'), 'w') as f:
#     json.dump(metrics, f)
#
# # ## 7. ROC Op
#
# # In[ ]:
#
#
# predictions = os.path.join(DATA_DIR, 'predict', 'prediction_results-*')
# target_lambda = """lambda x: 1 if (x['target'] > x['fare'] * 0.2) else 0"""
# true_class = 'true'
# true_score_column = 'true'
#
# if not os.path.exists(os.path.join(DATA_DIR, "roc")):
#     os.makedirs(os.path.join(DATA_DIR, "roc"))
#
# schema_file = os.path.join(os.path.dirname(predictions), 'schema.json')
# schema = json.loads(file_io.read_file_to_string(schema_file))
# names = [x['name'] for x in schema]
#
# if not target_lambda and 'target' not in names:
#     raise ValueError('There is no "target" column, and target_lambda is not provided.')
#
# if true_score_column not in names:
#     raise ValueError('Cannot find column name "%s"' % true_score_column)
#
# dfs = []
# files = file_io.get_matching_files(predictions)
# for file in files:
#     with file_io.FileIO(file, 'r') as f:
#         dfs.append(pd.read_csv(f, names=names))
#
# df = pd.concat(dfs)
# if target_lambda:
#     df['target'] = df.apply(eval(target_lambda), axis=1)
# else:
#     df['target'] = df['target'].apply(lambda x: 1 if x == true_class else 0)
# fpr, tpr, thresholds = roc_curve(df['target'], df[true_score_column])
# roc_auc = roc_auc_score(df['target'], df[true_score_column])
# df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
# roc_file = os.path.join(DATA_DIR + "roc", 'roc.csv')
# with file_io.FileIO(roc_file, 'w') as f:
#     df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False, index=False)
#
# metadata = {
#     'outputs': [{
#         'type': 'roc',
#         'format': 'csv',
#         'schema': [
#             {'name': 'fpr', 'type': 'NUMBER'},
#             {'name': 'tpr', 'type': 'NUMBER'},
#             {'name': 'thresholds', 'type': 'NUMBER'},
#         ],
#         'source': roc_file
#     }]
# }
# try:
#     os.makedirs(os.path.join(DATA_DIR, "artifacts", "roc"))
# except OSError:
#     pass
# with file_io.FileIO(os.path.join(DATA_DIR, 'artifacts', 'roc', 'mlpipeline-ui-metadata.json'), 'w') as f:
#     json.dump(metadata, f)
#
# metrics = {
#     'metrics': [{
#         'name': 'roc-auc-score',
#         'numberValue': roc_auc,
#     }]
# }
# with file_io.FileIO(os.path.join(DATA_DIR, 'artifacts', 'roc', 'mlpipeline-metrics.json'), 'w') as f:
#     json.dump(metrics, f)
#
# # ## 8. Deploy Op
#
# # In[2]:
#
#
# # pass: deploy would not be a "local" step
