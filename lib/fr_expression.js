import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import {softmax} from "./fr_pose";

async function loadExpressionModel() {
  var expression_session = null
  await InferenceSession.create("../model/fr_expression.onnx", {executionProviders: ['wasm']})
    .then((session) => {
      expression_session = session
      const input_tensor = new Tensor("float32", new Float32Array(224 * 224 * 3), [1, 3, 224, 224]);
      for (let i = 0; i < 224 * 224 * 3; i++) {
        input_tensor.data[i] = Math.random() * 2.0 - 1.0;
      }
      const feeds = {"input": input_tensor};
      const output_tensor = expression_session.run(feeds)
      console.log("initialize the expression session.")
    })
  return expression_session
}

function alignExpressionImage(image, bbox) {
  var src_h = image.rows,
    src_w = image.cols;

  var x = bbox[0]
  var y = bbox[1]
  var box_w = bbox[2]
  var box_h = bbox[3]

  var rect = new cv.Rect(x, y, Math.min(parseInt(box_w * 1.2), src_w - 1), Math.min(parseInt(box_h * 1.2), src_h - 1))

  var face_image = new cv.Mat()
  face_image = image.roi(rect)

  var dsize = new cv.Size(224, 224);
  var resize_image = new cv.Mat();
  cv.resize(face_image, resize_image, dsize);

  face_image.delete()
  return resize_image
}

function preprocessExpression(img) {
  var cols = img.cols;
  var rows = img.rows;
  var channels = 3;

  var img_data = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel = img.ucharPtr(y, x);
      for (var c = 0; c < channels; c++) {
        var pixel_value = pixel[c] / 255.0;
        img_data.set(y, x, c, pixel_value)
      }
    }

  var preprocesed = ndarray(new Float32Array(channels * cols * rows), [1, channels, rows, cols])

  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

async function predictExpression(session, canvas_id, bbox) {
  var img = cv.imread(canvas_id)

  var face_count = bbox.shape[0],
    bbox_size = bbox.shape[1];

  const result = [];
  for (let i = 0; i < face_count; i++) {
    var x1 = parseInt(bbox.data[i * bbox_size]),
      y1 = parseInt(bbox.data[i * bbox_size + 1]),
      x2 = parseInt(bbox.data[i * bbox_size + 2]),
      y2 = parseInt(bbox.data[i * bbox_size + 3]),
      width = Math.abs(x2 - x1),
      height = Math.abs(y2 - y1);

    var face_img = alignExpressionImage(img, [x1, y1, width, height]);
    //cv.imshow("live-temp", face_img);
    var input_image = preprocessExpression(face_img);
    face_img.delete();

    const input_tensor = new Tensor("float32", new Float32Array(224 * 224 * 3), [1, 3, 224, 224]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);
    const expression_arr = softmax(output_tensor['output'].data);

    var max_idx = null, max_val = 0;
    for (let i = 0; i < expression_arr.length; i++)
      if (max_val < expression_arr[i]) {
        max_idx = i;
        max_val = expression_arr[i];
      }
    result.push([x1, y1, x2, y2, max_idx]);
  }
  img.delete();
  return result;
}

export {loadExpressionModel, predictExpression}