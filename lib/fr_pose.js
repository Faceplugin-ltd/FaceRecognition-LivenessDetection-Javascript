import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";

async function loadPoseModel() {
  var pose_session = null
  await InferenceSession.create("../model/fr_pose.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        pose_session = session
        const input_tensor = new Tensor("float32", new Float32Array(224 * 224 * 3), [1, 3, 224, 224]);
        for (let i = 0; i < 224 * 224 * 3; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = pose_session.run(feeds)
        console.log("initialize the pose session.")
      })
  return pose_session
}

function preprocessPose(img) {
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

  var preprocesed = ndarray(new Float32Array(3 * 224 * 224), [1, 3, 224, 224])
  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

function softmax(arr) {
  return arr.map(function(value, index) {
    return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
  })
}

async function predictPose(session, canvas_id, bbox) {
  var face_count = bbox.shape[0],
      bbox_size = bbox.shape[1];

  var img = cv.imread(canvas_id);
  const result = [];

  for (let i = 0; i < face_count; i++) {
    var x1 = parseInt(bbox.data[i * bbox_size]),
        y1 = parseInt(bbox.data[i * bbox_size + 1]),
        x2 = parseInt(bbox.data[i * bbox_size + 2]),
        y2 = parseInt(bbox.data[i * bbox_size + 3]),
        width = Math.abs(x2 - x1),
        height = Math.abs(y2 - y1);

    var x11 = parseInt(x1 - width/4),
        y11 = parseInt(y1 - height/4),
        x22 = parseInt(x2 + width/4),
        y22 = parseInt(y2 + height/4);

    var rect = new cv.Rect(Math.max(x11, 0), Math.max(y11, 0), Math.min(x22 - x11, img.cols), Math.min(y22 - y11, img.rows));
    var face_image = new cv.Mat();

    face_image = img.roi(rect);

    var dsize = new cv.Size(224, 224);
    var resize_image = new cv.Mat();
    cv.resize(face_image, resize_image, dsize);
    cv.cvtColor(resize_image, resize_image, cv.COLOR_BGR2RGB);

    // cv.imshow("live-temp", resize_image)

    var input_image = preprocessPose(resize_image);

    const input_tensor = new Tensor("float32", new Float32Array(224 * 224 * 3), [1, 3, 224, 224]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);

    var arr = Array.apply(null, Array(66));
    const index_arr = arr.map(function (x, i) { return i })

    const yaw_arr = softmax(output_tensor['output'].data);
    const pitch_arr = softmax(output_tensor['617'].data);
    const roll_arr = softmax(output_tensor['618'].data);

    const yaw = yaw_arr.reduce(function (r, a, i){return r + a * index_arr[i]}, 0) * 3 - 99;
    const pitch = pitch_arr.reduce(function (r, a, i){return r + a * index_arr[i]}, 0) * 3 - 99;
    const roll = roll_arr.reduce(function (r, a, i){return r + a * index_arr[i]}, 0) * 3 - 99;
    //console.log("Pose results: ", yaw, pitch, roll)
    result.push([x1, y1, x2, y2, yaw.toFixed(2), pitch.toFixed(2), roll.toFixed(2)]);

    resize_image.delete();
    face_image.delete();
  }
  img.delete();
  return result;
}

export {loadPoseModel, softmax, predictPose}