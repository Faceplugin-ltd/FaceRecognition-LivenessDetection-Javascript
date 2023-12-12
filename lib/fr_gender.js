import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";

async function loadGenderModel() {
  var feature_session = null;
  await InferenceSession.create("../model/fr_gender.onnx", {executionProviders: ['wasm']})
    .then((session) => {
      feature_session = session
      const input_tensor = new Tensor("float32", new Float32Array(64 * 64 * 3), [1, 3, 64, 64]);
      for (let i = 0; i < 64 * 64 * 3; i++) {
        input_tensor.data[i] = Math.random() * 2.0 - 1.0;
      }
      const feeds = {"input": input_tensor};
      const output_tensor = feature_session.run(feeds)
      console.log("initialize the gender session.")
    })
  return feature_session;
}

function alignGenderImage(image, bbox, scale_value) {
  var src_h = image.rows,
    src_w = image.cols;

  var x = bbox[0]
  var y = bbox[1]
  var box_w = bbox[2]
  var box_h = bbox[3]

  var scale = Math.min((src_h - 1) / box_h, Math.min((src_w - 1) / box_w, scale_value))

  var new_width = box_w * scale
  var new_height = box_h * scale
  var center_x = box_w / 2 + x,
    center_y = box_h / 2 + y

  var left_top_x = center_x - new_width / 2
  var left_top_y = center_y - new_height / 2
  var right_bottom_x = center_x + new_width / 2
  var right_bottom_y = center_y + new_height / 2

  if (left_top_x < 0) {
    right_bottom_x -= left_top_x
    left_top_x = 0
  }

  if (left_top_y < 0) {
    right_bottom_y -= left_top_y
    left_top_y = 0
  }

  if (right_bottom_x > src_w - 1) {
    left_top_x -= right_bottom_x - src_w + 1
    right_bottom_x = src_w - 1
  }

  if (right_bottom_y > src_h - 1) {
    left_top_y -= right_bottom_y - src_h + 1
    right_bottom_y = src_h - 1
  }
  var rect = new cv.Rect(Math.max(parseInt(left_top_x), 0), Math.max(parseInt(left_top_y), 0),
    Math.min(parseInt(right_bottom_x - left_top_x), src_w - 1), Math.min(parseInt(right_bottom_y - left_top_y), src_h - 1))

  var face_image = new cv.Mat()
  face_image = image.roi(rect)

  var dsize = new cv.Size(64, 64);
  var resize_image = new cv.Mat();
  cv.resize(face_image, resize_image, dsize);

  face_image.delete()
  return resize_image
}

function mergeGender(x, s1, s2, s3, lambda_local, lambda_d) {
  let a = 0;
  let b = 0;
  let c = 0;

  const V = 1;

  for (let i = 0; i < s1; i++)
    a = a + (i + lambda_local * x[12 + i]) * x[i];
  // console.log("a = ", a)

  a = a / (s1 * (1 + lambda_d * x[9]));

  for (let i = 0; i < s2; i++)
    b = b + (i + lambda_local * x[15 + i]) * x[3 + i];
  //console.log("b = ", b)

  b = b / (s1 * (1 + lambda_d * x[9])) / (s2 * (1 + lambda_d * x[10]));

  for (let i = 0; i < s3; i++)
    c = c + (i + lambda_local * x[18 + i]) * x[6 + i];
  //console.log("c = ", c)

  c = c / (s1 * (1 + lambda_d * x[9])) / (s2 * (1 + lambda_d * x[10])) / (s3 * (1 + lambda_d * x[11]));
  return (a + b + c) * V;
}

function preprocessGender(img) {
  var cols = img.cols;
  var rows = img.rows;
  var channels = 3;

  var img_data = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel = img.ucharPtr(y, x);
      // if(x == 0 && y == 0)
      //     console.log(pixel);
      for (var c = 0; c < channels; c++) {
        var pixel_value = 0
        if (c === 0) // R
          pixel_value = (pixel[c] / 255.0 - 0.485) / 0.229
        if (c === 1) // G
          pixel_value = (pixel[c] / 255.0 - 0.456) / 0.224
        if (c === 2) // B
          pixel_value = (pixel[c] / 255.0 - 0.406) / 0.225

        img_data.set(y, x, c, pixel_value)
      }
    }

  var preprocesed = ndarray(new Float32Array(3 * 64 * 64), [1, 3, 64, 64])
  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

async function predictGender(session, canvas_id, bbox) {
  var img = cv.imread(canvas_id);

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

    var face_img = alignGenderImage(img, [x1, y1, width, height], 1.4);
    //cv.imshow("live-temp", face_img);
    var input_image = preprocessGender(face_img);
    face_img.delete();

    const input_tensor = new Tensor("float32", new Float32Array(64 * 64 * 3), [1, 3, 64, 64]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);

    const outputLayers = ["prob_stage_1", "prob_stage_2", "prob_stage_3", "stage1_delta_k", "stage2_delta_k", "stage3_delta_k",
                             "index_offset_stage1", "index_offset_stage2", "index_offset_stage3"];

    const outputFeat = [];
    for (let i = 0; i < outputLayers.length; i++) {
      const result = output_tensor[outputLayers[i]];
      // console.log(outputLayers[i], ": ", result.size);
      for (let j = 0; j < result.size; j++)
        outputFeat.push(result.data[j]);
    }

    // console.log("final result: ", outputFeat);
    let gender = mergeGender(outputFeat, 3, 3, 3, 1, 1);
    console.log("output gender: ", gender);
    result.push([x1, y1, x2, y2, gender]);
  }
  img.delete();
  return result;
}

export {loadGenderModel, predictGender}