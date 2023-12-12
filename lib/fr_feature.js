import {InferenceSession, Tensor} from "onnxruntime-web";
import ndarray from "ndarray";
import ops from "ndarray-ops";

const REFERENCE_FACIAL_POINTS = [
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014],
    [70.72990036, 92.20410156]
]

async function loadFeatureModel() {
  var feature_session = null;
  await InferenceSession.create("../model/fr_feature.onnx", {executionProviders: ['wasm']})
      .then((session) => {
        feature_session = session
        const input_tensor = new Tensor("float32", new Float32Array(112 * 112 * 3), [1, 3, 112, 112]);
        for (let i = 0; i < 112 * 112 * 3; i++) {
          input_tensor.data[i] = Math.random() * 2.0 - 1.0;
        }
        const feeds = {"input": input_tensor};
        const output_tensor = feature_session.run(feeds)
        console.log("initialize the feature session.")
      })
  return feature_session;
}

function convert68pts5pts(landmark) {
    var left_eye_x = (landmark[74] + landmark[76] + landmark[80] + landmark[82]) / 4,
        left_eye_y = (landmark[75] + landmark[77] + landmark[81] + landmark[83]) / 4,

        right_eye_x = (landmark[86] + landmark[88] + landmark[92] + landmark[94]) / 4,
        right_eye_y = (landmark[87] + landmark[89] + landmark[93] + landmark[95]) / 4,

        nose_x = landmark[60], nose_y = landmark[61],

        left_mouse_x = (landmark[96] + landmark[120]) / 2,
        left_mouse_y = (landmark[97] + landmark[121]) / 2,

        right_mouse_x = (landmark[108] + landmark[128]) / 2,
        right_mouse_y = (landmark[109] + landmark[129]) / 2;
    return [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y], [nose_x, nose_y], [left_mouse_x, left_mouse_y],
      [right_mouse_x, right_mouse_y]]
}

function getReferenceFacialPoints() {
  let ref5pts = REFERENCE_FACIAL_POINTS;

  return ref5pts;
}

function warpAndCropFace(src,
                       face_pts,
                       ref_pts=null,
                       crop_size=[112, 112]) {

  let srcTri = cv.matFromArray(3, 1, cv.CV_32FC2, [face_pts[0][0], face_pts[0][1], face_pts[1][0], face_pts[1][1],
    face_pts[2][0], face_pts[2][1]]);
  let dstTri = cv.matFromArray(3, 1, cv.CV_32FC2, [ref_pts[0][0], ref_pts[0][1], ref_pts[1][0], ref_pts[1][1],
    ref_pts[2][0], ref_pts[2][1]]);

  let tfm = cv.getAffineTransform(srcTri, dstTri);

  let dsize = new cv.Size(crop_size[0], crop_size[1]);
  let dst = new cv.Mat();
  cv.warpAffine(src, dst, tfm, dsize);

  return dst;
}

function alignFeatureImage(image, landmark) {
  let facePoints = convert68pts5pts(landmark);
  let refPoints = getReferenceFacialPoints();
  let alignImg = warpAndCropFace(image, facePoints, refPoints);
  return alignImg;
}

function preprocessFeature(image) {
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

  ops.assign(preprocesed.pick(0, 0, null, null), img_data.pick(null, null, 0));
  ops.assign(preprocesed.pick(0, 1, null, null), img_data.pick(null, null, 1));
  ops.assign(preprocesed.pick(0, 2, null, null), img_data.pick(null, null, 2));

  return preprocesed
}

async function extractFeatureImage(session, img, landmarks) {
  const result = [];
  for (let i = 0; i < landmarks.length; i++) {

    var face_img = alignFeatureImage(img, landmarks[i]);
    //cv.imshow("live-temp", face_img);
    var input_image = preprocessFeature(face_img);
    face_img.delete();

    const input_tensor = new Tensor("float32", new Float32Array(112 * 112 * 3), [1, 3, 112, 112]);
    input_tensor.data.set(input_image.data);
    const feeds = {"input": input_tensor};

    const output_tensor = await session.run(feeds);
    // console.log("Feature result: ", output_tensor);

    result.push(output_tensor);
  }
  img.delete();
  return result;
}

async function extractFeature(session, canvas_id, landmarks) {
  var img = cv.imread(canvas_id);

  var result = await extractFeatureImage(session, img, landmarks);
  return result;
}

async function extractFeatureBase64(session, base64Image, landmarks) {
  let image = new Image()
  image.src = base64Image;
  await new Promise(r => {
    image.onload = r
  })

  var img = cv.imread(image);

  var result = await extractFeatureImage(session, img, landmarks);
  return result;
}

function matchFeature(feature1, feature2) {
  const vectorSize = feature1.length;

  let meanFeat = [];
  let feature1Sum = 0
  let feature2Sum = 0

  for (let i = 0; i < vectorSize; i++) {
    let meanVal = (feature1[i] + feature2[i]) / 2;
    feature1[i] -= meanVal;
    feature2[i] -= meanVal;

    feature1Sum += feature1[i] * feature1[i];
    feature2Sum += feature2[i] * feature2[i];
  }

  let score = 0;
  for (let i = 0; i < vectorSize; i++) {
    feature1[i] = feature1[i] / Math.sqrt(feature1Sum);
    feature2[i] = feature2[i] / Math.sqrt(feature2Sum);

    score += feature1[i] * feature2[i];
  }

  return score
}

export {loadFeatureModel, extractFeature, extractFeatureBase64, matchFeature}