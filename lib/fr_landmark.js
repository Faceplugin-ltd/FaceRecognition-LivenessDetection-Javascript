import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";


async function loadLandmarkModel() {
  var landmark_session = null
  await InferenceSession.create("../model/fr_landmark.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        landmark_session = session
        const input_tensor = new Tensor("float32", new Float32Array(64 * 64), [1, 1, 64, 64]);
        for (let i = 0; i < 64 * 64; i++) {
          input_tensor.data[i] = Math.random();
        }
        const feeds = {"input": input_tensor};
        const output_tensor = landmark_session.run(feeds)
        console.log("initialize the landmark session.")
      })
  return landmark_session
}

function decodeLandmark(landmark, priors, variances) {

  var landms = ndarray(landmark.data, [26250, 10]);
  var before_prior = priors.hi(null, 2),
      after_prior = priors.lo(null, 2);
  var result = ndarray(new Float32Array(landms.shape[0] * landms.shape[1]), landms.shape);
  var priortemp = ndarray(new Float32Array(after_prior.shape[0] * 2), [after_prior.shape[0], 2]);
  var half_size = parseInt(Math.floor(landms.shape[1] / 2));

  ops.muls(priortemp, after_prior, variances[0]);

  for (var index = 0; index < half_size; index++) {
    let temp = ndarray(new Float32Array(landms.shape[0] * 2), [landms.shape[0], 2]);
    let temp2 = ndarray(new Float32Array(landms.shape[0] * 2), [landms.shape[0], 2]);
    let preslice = landms.hi(null, (index + 1) * 2).lo(null, index * 2);
    ops.mul(temp, preslice, priortemp);
    ops.add(temp2, temp, before_prior);
    ops.assign(result.pick(null, index * 2), temp2.pick(null, 0));
    ops.assign(result.pick(null, index * 2 + 1), temp2.pick(null, 1));
  }

  return result;
}

function alignLandmarkImage(image, bbox, scale_value) {
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

  var dsize = new cv.Size(64, 64);
  var resize_image = new cv.Mat();
  cv.resize(face_image, resize_image, dsize);
  cv.cvtColor(resize_image, resize_image, cv.COLOR_BGR2GRAY)

  face_image.delete()
  return resize_image
}

function preprocessLandmark(img) {
  var cols = img.cols;
  var rows = img.rows;
  var channels = 1;

  var img_data = ndarray(new Float32Array(rows * cols * channels), [rows, cols, channels]);

  for (var y = 0; y < rows; y++)
    for (var x = 0; x < cols; x++) {
      let pixel = img.ucharPtr(y, x);
      for (var c = 0; c < channels; c++) {
        var pixel_value = pixel[c] / 256.0;
        img_data.set(y, x, c, pixel_value)
      }
    }

  var preprocesed = ndarray(new Float32Array(channels * cols * rows), [1, channels, rows, cols])
  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));

  return preprocesed;
}

async function predictLandmarkImage(session, img, bbox) {
  var face_size = bbox.shape[0];
  var bbox_size = bbox.shape[1];

  const landmarks = [];

  for (let i = 0; i < face_size; i++) {
     var x1 = parseInt(bbox.data[i * bbox_size]),
        y1 = parseInt(bbox.data[i * bbox_size + 1]),
        x2 = parseInt(bbox.data[i * bbox_size + 2]),
        y2 = parseInt(bbox.data[i * bbox_size + 3]),
        width = Math.abs(x2 - x1),
        height = Math.abs(y2 - y1);

    var face_img = alignLandmarkImage(img, [x1, y1, width, height], 1.0);
    // cv.imshow("live-temp", face_img);
    var input_image = preprocessLandmark(face_img);
    face_img.delete();
    const input_tensor = new Tensor("float32", new Float32Array(64 * 64), [1, 1, 64, 64]);

    input_tensor.data.set(input_image.data);

    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);
    var landmark_arr = output_tensor['output'].data;

    for (let i = 0; i < landmark_arr.length; i++) {
      if (i % 2 === 0)
        landmark_arr[i] = parseInt(landmark_arr[i] * width + x1);
      else
        landmark_arr[i] = parseInt(landmark_arr[i] * height + y1);
    }
    landmarks.push(landmark_arr);
    // console.log("Landmark result: ", landmark_arr[0], landmark_arr[1], landmark_arr[74], landmark_arr[75], landmark_arr[76], landmark_arr[77]);
  }
  img.delete();
  return landmarks
}

async function predictLandmark(session, canvas_id, bbox) {

  var img = cv.imread(canvas_id);
  var landmarks = await predictLandmarkImage(session, img, bbox);

  return landmarks;
}

async function predictLandmarkBase64(session, base64Image, bbox) {
  let image = new Image()
  image.src = base64Image
  await new Promise(r => {
    image.onload = r
  })

  var img = cv.imread(image);
  var landmarks = await predictLandmarkImage(session, img, bbox);

  return landmarks;
}

export {loadLandmarkModel, predictLandmark, predictLandmarkBase64}