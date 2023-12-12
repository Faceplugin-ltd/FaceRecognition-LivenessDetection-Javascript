import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";
import {cv} from "./load_opencv";
import {download} from "./download";

async function loadDetectionModel() {
  var detect_session = null;
  await InferenceSession.create("../model/fr_detect.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        detect_session = session;
        const input_tensor = new Tensor("float32", new Float32Array(320 * 240 * 3), [1, 3, 240, 320]);
        for (let i = 0; i < 320 * 240 * 3; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = detect_session.run(feeds)
        console.log("initialize the detection session.")
      })
  return detect_session
}

async function loadDetectionModelPath(model_path) {
  const arr_buf = await download(model_path);
  const detection_session = await InferenceSession.create(arr_buf);
  return detection_session
}

function preprocessDetection(image) {
  var rows = image.rows,
      cols = image.cols;

  var img_data = ndarray(new Float32Array(rows * cols * 3), [rows, cols, 3]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel = image.ucharPtr(y, x);
      for (var c = 0; c < 3; c++) {
        var pixel_value = 0
        if (c === 0) // R
          pixel_value = (pixel[c] - 127) / 128.0;
        if (c === 1) // G
          pixel_value = (pixel[c] - 127) / 128.0;
        if (c === 2) // B
          pixel_value = (pixel[c] - 127) / 128.0;

        img_data.set(y, x, c, pixel_value)
      }
    }

  var preprocesed = ndarray(new Float32Array(3 * rows * cols), [1, 3, rows, cols])

  // Transpose
  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

async function detectFaceImage(session, img) {
  const onnx_config = {
        min_sizes: [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
        steps: [8, 16, 32, 64],
        variance: [0.1, 0.2],
        clip: false,
        confidence_threshold: 0.65,
        top_k: 750,
        nms_threshold: 0.4,
      };

  var dsize = new cv.Size(320, 240);
  var resize_image = new cv.Mat();
  cv.resize(img, resize_image, dsize);
  cv.cvtColor(resize_image, resize_image, cv.COLOR_BGR2RGB);

  const image = preprocessDetection(resize_image);

  var resize_param = {cols: img.cols / 320, rows: img.rows / 240};

  const input_tensor = new Tensor("float32", new Float32Array(320 * 240 * 3), [1, 3, 240, 320]);
  input_tensor.data.set(image.data);

  const feeds = {"input": input_tensor};
  const output_tensor = await session.run(feeds);

  const loc = output_tensor['boxes'];
  const conf = output_tensor['scores'];

  // const landmarks = output_tensor['585']
  const total_result = conf.size / 2;

  const scale = [320, 240, 320, 240];
  const scale1 = [800, 800, 800, 800, 800, 800, 800, 800, 800, 800];

  const priors = definePriorBox([320, 240], onnx_config);

  const boxes_arr = decodeBBox(loc, priors, onnx_config);

  const scores_arr = ndarray(conf.data, [total_result, 2]).pick(null, 1);
  var landms_arr = null;//decode_landmark(landmarks, priors, onnx_config.variance);

  var box = ndarray(loc.data, [4420, 4]);
  var boxes_before = scaleMultiplyBBox(box, scale);
  var landms_before = null;//scale_multiply_landms(landms_arr, scale1);

  var [bbox_screen, scores_screen, landms_screen] = screenScore(boxes_before, scores_arr, landms_before, onnx_config.confidence_threshold);

  var [bbox_sorted, scores_sorted, landms_sorted] = sortScore(bbox_screen, scores_screen, landms_screen, onnx_config.top_k);

  var [bbox_small, score_result, landms_small, result_size] = cpuNMS(bbox_sorted, scores_sorted, landms_sorted, onnx_config.nms_threshold);
  var [bbox_result, landms_result] = scaleResult(bbox_small, landms_small, resize_param, img.cols, img.rows);

  var output = {
    bbox: bbox_result,
    landmark: landms_result,
    conf: score_result,
    size: result_size
  }

  resize_image.delete();
  img.delete();
  return output
}

async function detectFace(session, canvas_id) {

  var img = cv.imread(canvas_id);
  var output = await detectFaceImage(session, img);

  return output
}

async function detectFaceBase64(session, base64Image) {
  let image = new Image()
  image.src = base64Image
  await new Promise(r => {
    image.onload = r
  })

  var img = cv.imread(image);
  var output = await detectFaceImage(session, img);

  return output
}

function product(x, y) {
  var size_x = x.length,
      size_y = y.length;
  var result = [];

  for (var i = 0; i < size_x; i++)
    for (var j = 0; j < size_y; j++)
      result.push([x[i], y[j]]);

  return result;
}

function range(num) {
  var result = [];
  for (var i = 0; i < num; i++)
    result.push(i);

  return result;
}

function definePriorBox(image_size, onnx_config) {
  var min_sizes = onnx_config.min_sizes,
      steps = onnx_config.steps,
      clip = onnx_config.clip,
      name = "s",
      feature_maps = steps.map((step) => [Math.ceil(image_size[0] / step), Math.ceil(image_size[1] / step)]);

  var anchors = [];

  feature_maps.forEach((f, k) => {
    var min_size = min_sizes[k];
    product(range(f[0]), range(f[1])).forEach(([i, j]) => {
      min_size.forEach((m_size) => {
        var s_kx = m_size / image_size[0],
            s_ky = m_size / image_size[1];
        var dense_cx = [j + 0.5].map((x) => x * steps[k] / image_size[0]),
            dense_cy = [i + 0.5].map((y) => y * steps[k] / image_size[1]);
        product(dense_cy, dense_cx).forEach(([cy, cx]) => {
          anchors.push(cx);
          anchors.push(cy);
          anchors.push(s_kx);
          anchors.push(s_ky);
        })
      });
    });
  });

  var output = ndarray(new Float32Array(anchors), [anchors.length / 4, 4]);

  if (clip)
    output = ndarray.ops.min(1, ops.max(output, 0));

  return output;
}

function decodeBBox(bbox, priors, onnx_config) {
  var variances = onnx_config.variance
  var loc = ndarray(bbox.data, [4420, 4]);
  // console.log(bbox, priors);
  var before_prior = priors.hi(null, 2),
      after_prior = priors.lo(null, 2);

  var before_loc = loc.hi(null, 2),
      after_loc = loc.lo(null, 2);

  var before_result = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var before_temp = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var before_temp2 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);

  var after_result = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var after_temp = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var after_temp2 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var after_temp3 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);
  var after_temp4 = ndarray(new Float32Array(before_loc.shape[0] * before_loc.shape[1]), [before_loc.shape[0], 2]);

  var boxes = ndarray(new Float32Array(before_loc.shape[0] * 4), [before_loc.shape[0], 4]);

  // Before
  ops.mul(before_temp, before_loc, after_prior);
  ops.muls(before_temp2, before_temp, variances[0]);
  ops.add(before_result, before_temp2, before_prior);

  // After
  ops.muls(after_temp, after_loc, variances[1]);
  ops.exp(after_temp2, after_temp);
  ops.mul(after_temp3, after_temp2, after_prior);

  for (var index = 0; index < 4; index++)
    ops.assign(after_result.pick(null, index), after_temp3.pick(null, index));

  ops.divs(after_temp4, after_temp3, -2);
  ops.addeq(before_result, after_temp4);

  ops.addeq(after_result, before_result);

  ops.assign(boxes.pick(null, 0), before_result.pick(null, 0));
  ops.assign(boxes.pick(null, 1), before_result.pick(null, 1));
  ops.assign(boxes.pick(null, 2), after_result.pick(null, 0));
  ops.assign(boxes.pick(null, 3), after_result.pick(null, 1));

  return boxes;
}

function scaleMultiplyBBox(boxes_arr, scale) {
  var total_result = boxes_arr.shape[0];
  var boxes_before = ndarray(new Float32Array(total_result * 4), [total_result, 4]);

  for (var index = 0; index < scale.length; index++) {
    let temp = boxes_arr.pick(null, index),
        before_result = ndarray(new Float32Array(total_result), [total_result]);
    ops.muls(before_result, temp, scale[index]);
    ops.assign(boxes_before.pick(null, index), before_result);
  }

  return boxes_before;
}

function scaleMultiplyLandmarks(landms_arr, scale1) {
  var total_result = landms_arr.shape[0];
  var landms_before = ndarray(new Float32Array(total_result * 10), [total_result, 10]);

  for (var index = 0; index < scale1.length; index++) {
    let temp = landms_arr.pick(null, index),
        before_landms_result = ndarray(new Float32Array(total_result), [total_result]);
    ops.muls(before_landms_result, temp, scale1[index]);
    ops.assign(landms_before.pick(null, index), before_landms_result);
  }

  return landms_before;
}

function screenScore(bbox, scores, landms, threshold) {
  var total_size = scores.shape[0];
  var index_arr = [];

  for (var index = 0; index < total_size; index++) {
    var score_temp = scores.get(index);

    if (score_temp >= threshold) {
      index_arr.push(index);
    }
  }

  var result_bbox = ndarray(new Float32Array(index_arr.length * 4), [index_arr.length, 4]);
  var result_scores = ndarray(new Float32Array(index_arr.length), [index_arr.length]);
  var result_landms = null;//ndarray(new Float32Array(index_arr.length * 10), [index_arr.length, 10]);

  index_arr.forEach((index, i) => {
    ops.assign(result_bbox.pick(i, null), bbox.pick(index, null));
    //ops.assign(result_landms.pick(i, null), landms.pick(index, null));
    ops.assign(result_scores.pick(i), scores.pick(index));
  });

  return [result_bbox, result_scores, result_landms];
}

function sortScore(bbox, scores, landms, top_k) {
  var total_size = scores.shape[0];
  var index_sort = new Array(total_size * 2);

  for (var index = 0; index < total_size; index++) {
    var temp = scores.get(index);
    index_sort[index] = [index, temp];
  }

  index_sort.sort((a, b) => {
    if (a[1] < b[1]) return 1;
    if (a[1] > b[1]) return -1;

    return 0;
  });

  var max_size = (total_size > top_k) ? top_k : total_size;

  var result_bbox = ndarray(new Float32Array(max_size * 4), [max_size, 4]);
  var result_scores = ndarray(new Float32Array(max_size), [max_size]);
  var result_landms = null;//ndarray(new Float32Array(max_size * 10), [max_size, 10]);

  for (var idx = 0; idx < max_size; idx++) {
    result_scores.set(idx, index_sort[idx][1]);
    ops.assign(result_bbox.pick(idx, null), bbox.pick(index_sort[idx][0], null));
    //ops.assign(result_landms.pick(idx, null), landms.pick(index_sort[idx][0], null));
  }

  return [result_bbox, result_scores, result_landms];
}

function cpuNMS(bbox, scores, landms, thresh) {
  var {max, min} = Math;
  var size = bbox.shape[0];
  var foundLocations = [];
  var pick = [];

  for (var i = 0; i < size; i++) {
    var x1 = bbox.get(i, 0),
        y1 = bbox.get(i, 1),
        x2 = bbox.get(i, 2),
        y2 = bbox.get(i, 3);

    var width = x2 - x1,
        height = y2 - y1;

    if (width > 0 && height > 0) {
      var area = width * height;
      foundLocations.push({x1, y1, x2, y2, width, height, area, index: i});
    }
  }

  foundLocations.sort((b1, b2) => {
    return b1.y2 - b2.y2;
  });

  while (foundLocations.length > 0) {
    var last = foundLocations[0] //[foundLocations.length - 1];
    var suppress = [last];
    pick.push(last.index) //foundLocations.length - 1);

    for (let i = 1; i < foundLocations.length; i++) {
      const box = foundLocations[i];
      const xx1 = max(box.x1, last.x1);
      const yy1 = max(box.y1, last.y1);
      const xx2 = min(box.x2, last.x2);
      const yy2 = min(box.y2, last.y2);
      const w = max(0, xx2 - xx1 + 1);
      const h = max(0, yy2 - yy1 + 1);
      const overlap = (w * h) / box.area;

      if (overlap >= thresh)
        suppress.push(foundLocations[i]);
    }

    foundLocations = foundLocations.filter((box) => {
      return !suppress.find((supp) => {
        return supp === box;
      })
    });
  }

  var result_bbox = ndarray(new Float32Array(pick.length * 4), [pick.length, 4]);
  var result_scores = ndarray(new Float32Array(pick.length), [pick.length]);
  var result_landms = null;//ndarray(new Float32Array(pick.length * 10), [pick.length, 10]);

  // console.log("Pick index: ", pick);

  pick.forEach((pick_index, i) => {
    ops.assign(result_bbox.pick(i, null), bbox.pick(pick_index, null));
    ops.assign(result_scores.pick(i), scores.pick(pick_index));
    //ops.assign(result_landms.pick(i, null), landms.pick(pick_index, null));
  });

  return [result_bbox, result_scores, result_landms, pick.length];
}

function scaleResult(bbox, landmark, resize_param, width, height) {
  var size = bbox.shape[0];
  var result_bbox = ndarray(new Float32Array(size * 4), [size, 4]);
  var result_landms = null;//ndarray(new Float32Array(size * 10), [size, 10]);

  for (let i = 0; i < size; i++) {
    let x1 = bbox.get(i, 0) * resize_param.cols,
        y1 = bbox.get(i, 1) * resize_param.rows,
        x2 = bbox.get(i, 2) * resize_param.cols,
        y2 = bbox.get(i, 3) * resize_param.rows;

    const f_size = (y2 - y1);
    const ct_x = (x1 + x2) / 2,
          ct_y = (y1 + y2) / 2;

    x1 = (ct_x - f_size / 2) < 0 ? 0 : (ct_x - f_size / 2);
    y1 = (ct_y - f_size / 2) < 0 ? 0 : (ct_y - f_size / 2);
    x2 = (ct_x + f_size / 2) > width - 1 ? width - 1 : (ct_x + f_size / 2);
    y2 = (ct_y + f_size / 2) > height - 1 ? height - 1 : (ct_y + f_size / 2);

    result_bbox.set(i, 0, x1);
    result_bbox.set(i, 1, y1);
    result_bbox.set(i, 2, x2);
    result_bbox.set(i, 3, y2);
  }

  /*
  for (var j = 0; j < 5; j++) {
    let x = landmark.pick(null, j * 2),
        y = landmark.pick(null, j * 2 + 1);

    ops.divseq(x, resize_param.cols);     // X
    ops.divseq(y, resize_param.rows);    // Y

    ops.assign(result_landms.pick(null, j * 2), x);
    ops.assign(result_landms.pick(null, j * 2 + 1), y);
  }
  */
  return [result_bbox, result_landms];
}

function getBestFace(bbox) {
  var face_count = bbox.shape[0],
      bbox_size = bbox.shape[1];

  var idx = -1, max_size = 0;
  for (let i = 0; i < face_count; i++) {
    var x1 = parseInt(bbox.data[i * bbox_size]),
        y1 = parseInt(bbox.data[i * bbox_size + 1]),
        x2 = parseInt(bbox.data[i * bbox_size + 2]),
        y2 = parseInt(bbox.data[i * bbox_size + 3]),
        width = Math.abs(x2 - x1),
        height = Math.abs(y2 - y1);
    if (width * height > max_size)
      idx = i;
  }
  return idx
}

export {loadDetectionModel, loadDetectionModelPath, detectFace, detectFaceBase64}