import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import {softmax} from "./fr_pose";

async function loadEyeModel() {
  var eye_session = null
  await InferenceSession.create("../model/fr_eye.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        eye_session = session
        const input_tensor = new Tensor("float32", new Float32Array(24 * 24 * 1), [1, 1, 24, 24]);
        for (let i = 0; i < 24 * 24; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = eye_session.run(feeds)
        console.log("initialize the eye session.")
      })
  return eye_session
}

function getEyeBBox(landmark, size) {
  var height = size[0], width = size[1];
  const padding_rate = 1.6;
  var left_eye_center_x = parseInt((landmark[74] + landmark[76] + landmark[80] + landmark[82]) / 4);
  var left_eye_center_y = parseInt((landmark[75] + landmark[77] + landmark[81] + landmark[83]) / 4);
  var left_eye_size = parseInt((landmark[78] - landmark[72]) * padding_rate);
  var left_corner_x = parseInt(left_eye_center_x - left_eye_size / 2);
  if (left_corner_x < 0)
    left_corner_x = 0;

  var left_corner_y = parseInt(left_eye_center_y - left_eye_size / 2);
  if (left_corner_y < 0)
    left_corner_y = 0;

  if (left_corner_x + left_eye_size >= width)
    left_eye_size = width - left_corner_x - 1

  if (left_corner_y + left_eye_size >= height)
    left_eye_size = height - left_corner_y - 1

  var right_eye_center_x = parseInt((landmark[86] + landmark[88] + landmark[92] + landmark[94]) / 4);
  var right_eye_center_y = parseInt((landmark[87] + landmark[89] + landmark[93] + landmark[95]) / 4);
  var right_eye_size = parseInt((landmark[90] - landmark[84]) * padding_rate);
  var right_corner_x = parseInt(right_eye_center_x - right_eye_size / 2);
  if (right_corner_x < 0)
      right_corner_x = 0
  var right_corner_y = parseInt(right_eye_center_y - right_eye_size / 2);
  if (right_corner_y < 0)
    right_corner_y = 0
  if (right_corner_x + right_eye_size >= width)
      right_eye_size = width - right_corner_x - 1
  if (right_corner_y + right_eye_size >= height)
      right_eye_size = height - right_corner_y - 1

  return [left_corner_x, left_corner_y, left_eye_size, left_eye_size,
    right_corner_x, right_corner_y, right_eye_size, right_eye_size]
}

function alignEyeImage(image, landmark) {
  var src_h = image.rows,
      src_w = image.cols;

  var eye_bbox = getEyeBBox(landmark, [src_h, src_w])
  var rect = new cv.Rect(eye_bbox[0], eye_bbox[1], eye_bbox[2], eye_bbox[3])

  var eye_image = new cv.Mat()
  eye_image = image.roi(rect)

  var dsize = new cv.Size(24, 24);
  var left_eye = new cv.Mat();
  cv.resize(eye_image, left_eye, dsize);
  cv.cvtColor(left_eye, left_eye, cv.COLOR_BGR2GRAY)

  // right eye
  rect = new cv.Rect(eye_bbox[4], eye_bbox[5], eye_bbox[6], eye_bbox[7])
  eye_image = image.roi(rect)
  var right_eye = new cv.Mat();
  cv.resize(eye_image, right_eye, dsize);
  cv.cvtColor(right_eye, right_eye, cv.COLOR_BGR2GRAY)

  eye_image.delete()
  return [left_eye, right_eye]
}

function preprocessEye(imgs) {
  var cols = imgs[0].cols;
  var rows = imgs[0].rows;
  var channels = 1;

  var img_data1 = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);
  var img_data2 = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel1 = imgs[0].ucharPtr(y, x);
      let pixel2 = imgs[1].ucharPtr(y, x);

      for (var c = 0; c < channels; c++) {
        var pixel_value1 = pixel1[c] / 255.0;
        var pixel_value2 = pixel2[c] / 255.0;

        img_data1.set(y, x, c, pixel_value1)
        img_data2.set(y, x, c, pixel_value2)
      }
    }

  var preprocesed1 = ndarray(new Float32Array(channels * cols * rows), [1, channels, rows, cols])
  ops.assign(preprocesed1.pick(0, 0, null, null), img_data1.pick(null, null, 0));
  var preprocesed2 = ndarray(new Float32Array(channels * cols * rows), [1, channels, rows, cols])
  ops.assign(preprocesed2.pick(0, 0, null, null), img_data2.pick(null, null, 0));

  return [preprocesed1, preprocesed2]
}

async function predictEye(session, canvas_id, landmarks) {
  var img = cv.imread(canvas_id)

  const result = [];
  for (let i = 0; i < landmarks.length; i++) {
    var face_imgs = alignEyeImage(img, landmarks[i]);
    // cv.imshow("live-temp", face_imgs[1]);
    var input_images = preprocessEye(face_imgs);
    face_imgs[0].delete();
    face_imgs[1].delete();

    const input_tensor1 = new Tensor("float32", new Float32Array(24 * 24), [1, 1, 24, 24]);
    input_tensor1.data.set(input_images[0].data);
    const feeds1 = {"input": input_tensor1};

    const output_tensor1 = await session.run(feeds1);
    const left_res = softmax(output_tensor1['output'].data);

    const input_tensor2 = new Tensor("float32", new Float32Array(24 * 24), [1, 1, 24, 24]);
    input_tensor2.data.set(input_images[1].data);
    const feeds2 = {"input": input_tensor2};

    const output_tensor2 = await session.run(feeds2);
    const right_res = softmax(output_tensor2['output'].data);
    result.push([left_res[0] > left_res[1], right_res[0] > right_res[1]]);
  }
  img.delete();
  return result;
}

export {loadEyeModel, predictEye}