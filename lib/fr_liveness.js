import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import {softmax} from "./fr_pose";

async function loadLivenessModel() {
  var live_session = null
  await InferenceSession.create("../model/fr_liveness.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        live_session = session
        const input_tensor = new Tensor("float32", new Float32Array(128 * 128 * 3), [1, 3, 128, 128]);
        for (let i = 0; i < 128 * 128 * 3; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = live_session.run(feeds)
        console.log("initialize the live session.")
      })
  return live_session;
}

function alignLivenessImage(image, bbox, scale_value) {
  var src_h = image.rows,
      src_w = image.cols;

  var x = bbox[0]
  var y = bbox[1]
  var box_w = bbox[2]
  var box_h = bbox[3]

  var scale = Math.min((src_h-1)/box_h, Math.min((src_w-1)/box_w, scale_value))

  var new_width = box_w * scale
  var new_height = box_h * scale
  var center_x = box_w/2+x,
      center_y = box_h/2+y

  var left_top_x = center_x-new_width/2
  var left_top_y = center_y-new_height/2
  var right_bottom_x = center_x+new_width/2
  var right_bottom_y = center_y+new_height/2

  if (left_top_x < 0) {
    right_bottom_x -= left_top_x
    left_top_x = 0
  }

  if (left_top_y < 0) {
    right_bottom_y -= left_top_y
    left_top_y = 0
  }

  if (right_bottom_x > src_w-1) {
    left_top_x -= right_bottom_x-src_w+1
    right_bottom_x = src_w-1
  }

  if (right_bottom_y > src_h-1) {
    left_top_y -= right_bottom_y-src_h+1
    right_bottom_y = src_h-1
  }
  var rect = new cv.Rect(Math.max(parseInt(left_top_x), 0), Math.max(parseInt(left_top_y), 0),
      Math.min(parseInt(right_bottom_x - left_top_x), src_w-1), Math.min(parseInt(right_bottom_y - left_top_y), src_h-1))

  var face_image = new cv.Mat()
  face_image = image.roi(rect)

  var dsize = new cv.Size(128, 128);
  var resize_image = new cv.Mat();
  cv.resize(face_image, resize_image, dsize);

  face_image.delete()
  return resize_image
}

function preprocessLiveness(img) {
  var cols = img.cols;
  var rows = img.rows;
  var channels = 3;

  var img_data = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel = img.ucharPtr(y, x);
      for (var c = 0; c < channels; c++) {
        var pixel_value = 0
        if (c === 0) // R
          pixel_value = pixel[c];
        if (c === 1) // G
          pixel_value = pixel[c];
        if (c === 2) // B
          pixel_value = pixel[c];

        img_data.set(y, x, c, pixel_value)
      }
    }

  var preprocesed = ndarray(new Float32Array(channels * cols * rows), [1, channels, rows, cols])
  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

async function predictLiveness(session, canvas_id, bbox) {
  var img = cv.imread(canvas_id)

  var face_size = bbox.shape[0];
  var bbox_size = bbox.shape[1];

  const result = [];
  for (let i = 0; i < face_size; i++) {
    var x1 = parseInt(bbox.data[i * bbox_size]),
        y1 = parseInt(bbox.data[i * bbox_size + 1]),
        x2 = parseInt(bbox.data[i * bbox_size + 2]),
        y2 = parseInt(bbox.data[i * bbox_size + 3]),
        width = Math.abs(x2 - x1),
        height = Math.abs(y2 - y1);

    var face_img = alignLivenessImage(img, [x1, y1, width, height], 2.7);
    //cv.imshow("live-temp", face_img);
    var input_image = preprocessLiveness(face_img);
    face_img.delete();

    const input_tensor = new Tensor("float32", new Float32Array(128 * 128 * 3), [1, 3, 128, 128]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);
    const score_arr = softmax(output_tensor['output'].data);
    console.log("Liveness result: ", score_arr);

    result.push([x1, y1, x2, y2, score_arr[0]]);
  }
  img.delete();
  return result;

}

async function predictLivenessBase64(session, base64Image) {
  let image = new Image()
  image.src = base64Image
  await new Promise(r => {
    image.onload = r
  })

  var img = cv.imread(image)

  var face_size = bbox.shape[0];
  var bbox_size = bbox.shape[1];

  const result = [];
  for (let i = 0; i < face_size; i++) {
    var x1 = parseInt(bbox.data[i * bbox_size]),
        y1 = parseInt(bbox.data[i * bbox_size + 1]),
        x2 = parseInt(bbox.data[i * bbox_size + 2]),
        y2 = parseInt(bbox.data[i * bbox_size + 3]),
        width = Math.abs(x2 - x1),
        height = Math.abs(y2 - y1);

    var face_img = alignLivenessImage(img, [x1, y1, width, height], 2.7);
    //cv.imshow("live-temp", face_img);
    var input_image = preprocessLiveness(face_img);
    face_img.delete();

    const input_tensor = new Tensor("float32", new Float32Array(128 * 128 * 3), [1, 3, 128, 128]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);
    const score_arr = softmax(output_tensor['output'].data);
    console.log("Liveness result: ", score_arr);

    result.push([x1, y1, x2, y2, score_arr[0]]);
  }
  img.delete();
  return result;

}

export {loadLivenessModel, predictLiveness}
