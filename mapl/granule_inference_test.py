# Copyright 2026 Google LLC.
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

"""Unit tests for the EMIT granule strided batched inference runner."""

from absl.testing import absltest
from mapl import granule_inference
import mapl.data_types
import numpy as np
import tensorflow as tf


class MockInferFn:
  """Mocks an inference function, returning dummy outputs."""

  def __init__(self, with_signature=False):
    if with_signature:
      # define input_signature matching what the code expects
      self.input_signature = [{
          'emit_l1b_radiance': tf.TensorSpec(
              shape=(None, None, None, None), dtype=tf.float32
          ),
          'emit_l1b_to_sun_zenith': tf.TensorSpec(
              shape=(None, None, None), dtype=tf.float32
          ),
          'emit_l1b_to_sensor_zenith': tf.TensorSpec(
              shape=(None, None, None), dtype=tf.float32
          ),
          'emit_l1b_crosstrack_id': tf.TensorSpec(
              shape=(None, None, None), dtype=tf.int32
          ),
      }]
    else:
      self.input_signature = None

  def __call__(self, inputs):
    # Expect inputs batched with shape (batch=1, h, w, ...)
    h_shape = inputs['emit_l1b_radiance'].shape[1]
    w_shape = inputs['emit_l1b_radiance'].shape[2]

    concentration = tf.ones((1, h_shape, w_shape, 1))
    binary_masks = tf.ones((1, h_shape, w_shape, 1))
    origin_masks = tf.ones((1, h_shape, w_shape, 1))

    return {
        'concentration': concentration,
        'binary_masks': binary_masks,
        'origin_masks': origin_masks,
    }


class MockModel:
  """Mock model for testing tiled inference."""

  def __init__(self, with_signature=False):
    self.infer_fn = MockInferFn(with_signature)


class GranuleInferenceTest(absltest.TestCase):

  def test_epsg_to_utm(self):
    self.assertEqual(granule_inference._epsg_to_utm('EPSG:32631'), '31N')
    self.assertEqual(granule_inference._epsg_to_utm('EPSG:32760'), '60S')
    with self.assertRaises(ValueError):
      granule_inference._epsg_to_utm('EPSG:4326')

  def test_crop_and_pad_2d(self):
    arr = np.ones((10, 10))

    # Normal crop
    cropped = granule_inference._crop_and_pad(arr, 2, 8, 2, 8)
    self.assertEqual(cropped.shape, (6, 6))
    self.assertEqual(np.sum(cropped), 36)

    # Pad before
    padded = granule_inference._crop_and_pad(arr, -2, 8, -2, 8)
    self.assertEqual(padded.shape, (10, 10))
    self.assertEqual(np.sum(padded), 64)  # 8x8 area inside
    self.assertEqual(padded[0, 0], 0)

    # Pad after
    padded = granule_inference._crop_and_pad(arr, 2, 12, 2, 12)
    self.assertEqual(padded.shape, (10, 10))
    self.assertEqual(np.sum(padded), 64)
    self.assertEqual(padded[-1, -1], 0)

  def test_crop_and_pad_3d(self):
    arr = np.ones((10, 10, 3))
    cropped = granule_inference._crop_and_pad(arr, 2, 8, 2, 8)
    self.assertEqual(cropped.shape, (6, 6, 3))

    padded = granule_inference._crop_and_pad(arr, -2, 8, -2, 8)
    self.assertEqual(padded.shape, (10, 10, 3))
    self.assertEqual(np.sum(padded), 64 * 3)

  def test_run_tiled_inference(self):
    model = MockModel(with_signature=False)

    h, w, c = 100, 100, 3
    l1b_radiance = np.ones((h, w, c), dtype=np.float32)
    sun_zenith = np.ones((h, w), dtype=np.float32)
    sensor_zenith = np.ones((h, w), dtype=np.float32)
    crosstrack_ids = np.ones((h, w), dtype=np.int32)
    scene_mask = np.ones((h, w), dtype=np.int32)

    granule = mapl.data_types.Granule(
        ee_asset_id='test',
        data={
            'emit_l1b_radiance': l1b_radiance,
            'emit_l1b_to_sun_zenith': sun_zenith,
            'emit_l1b_to_sensor_zenith': sensor_zenith,
            'emit_l1b_crosstrack_id': crosstrack_ids,
        },
        mask=scene_mask,
        geotransform=(0, 1, 0, 0, 0, -1),
        utm_zone='31N',
        epsg='EPSG:32631',
        timestamp_ms=0,
    )

    input_size = 64
    stride = 32

    chunks = list(
        granule_inference.run_tiled_inference(
            granule=granule,
            infer_fn=model.infer_fn,
            input_size=input_size,
            stride=stride,
            batch_size=1,
            batch_preprocessing_fn=None,
            outputs_names=None,
        )
    )

    # For h=100, w=100, stride=32, input_size=64
    # 100 - 64 + 1 = 37. arange(0, 37, 32) -> [0, 32]
    # So 2x2 = 4 chunks
    self.assertLen(chunks, 4)

    # Check the first chunk
    chunk = chunks[0]
    self.assertEqual(chunk.h_off, 0)
    self.assertEqual(chunk.w_off, 0)
    self.assertEqual(chunk.pad_y, 0)
    self.assertEqual(chunk.pad_x, 0)

    # The output shapes should match the input patch size (which is input_size)
    self.assertEqual(
        chunk.data['concentration'].shape, (input_size, input_size, 1)
    )

  def test_run_tiled_inference_with_signature(self):
    model = MockModel(with_signature=True)

    h, w, c = 64, 64, 3
    l1b_radiance = np.ones((h, w, c), dtype=np.float32)
    sun_zenith = np.ones((h, w), dtype=np.float32)
    sensor_zenith = np.ones((h, w), dtype=np.float32)
    crosstrack_ids = np.ones((h, w), dtype=np.int32)
    scene_mask = np.ones((h, w), dtype=np.int32)

    granule = mapl.data_types.Granule(
        ee_asset_id='test',
        data={
            'emit_l1b_radiance': l1b_radiance,
            'emit_l1b_to_sun_zenith': sun_zenith,
            'emit_l1b_to_sensor_zenith': sensor_zenith,
            'emit_l1b_crosstrack_id': crosstrack_ids,
        },
        mask=scene_mask,
        geotransform=(0, 1, 0, 0, 0, -1),
        utm_zone='31N',
        epsg='EPSG:32631',
        timestamp_ms=0,
    )

    input_size = 64
    stride = 64

    chunks = list(
        granule_inference.run_tiled_inference(
            granule=granule,
            infer_fn=model.infer_fn,
            input_size=input_size,
            stride=stride,
            batch_size=1,
            batch_preprocessing_fn=None,
            outputs_names=None,
        )
    )

    self.assertLen(chunks, 1)

  def test_run_tiled_inference_masking(self):
    model = MockModel(with_signature=False)

    h, w, c = 64, 64, 3
    l1b_radiance = np.ones((h, w, c), dtype=np.float32)
    sun_zenith = np.ones((h, w), dtype=np.float32)
    sensor_zenith = np.ones((h, w), dtype=np.float32)
    crosstrack_ids = np.ones((h, w), dtype=np.int32)
    # Mask is completely zero, which means inference should skip this area
    scene_mask = np.zeros((h, w), dtype=np.int32)

    granule = mapl.data_types.Granule(
        ee_asset_id='test',
        data={
            'emit_l1b_radiance': l1b_radiance,
            'emit_l1b_to_sun_zenith': sun_zenith,
            'emit_l1b_to_sensor_zenith': sensor_zenith,
            'emit_l1b_crosstrack_id': crosstrack_ids,
        },
        mask=scene_mask,
        geotransform=(0, 1, 0, 0, 0, -1),
        utm_zone='31N',
        epsg='EPSG:32631',
        timestamp_ms=0,
    )

    input_size = 64
    stride = 64

    chunks = list(
        granule_inference.run_tiled_inference(
            granule=granule,
            infer_fn=model.infer_fn,
            input_size=input_size,
            stride=stride,
            batch_size=1,
            batch_preprocessing_fn=None,
            outputs_names=None,
        )
    )

    self.assertEmpty(chunks)


if __name__ == '__main__':
  absltest.main()
