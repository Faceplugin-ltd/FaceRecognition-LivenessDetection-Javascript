<div align="center">
<img alt="" src="https://github.com/Faceplugin-ltd/FaceRecognition-Javascript/assets/160750757/657130a9-50f2-486d-b6d5-b78bcec5e6e2.png" width=200/>
</div>


<div align="left">
<img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" alt="Awesome Badge"/>
<img src="https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99" alt="Star Badge"/>
<img src="https://img.shields.io/github/issues/genderev/assassin" alt="issue"/>
<img src="https://img.shields.io/github/issues-pr/genderev/assassin" alt="pr"/>
</div>

<h2>Installation</h2>

```bash
npm install faceplugin
```

<h2>Table of Contents</h2>

* **[Face Detection](#face-detection)**
* **[Face Landmark Extraction](#face-landmark-extraction)**
* **[Face Liveness Detection](#face-expression-detection)**
* **[Face Expression Detection](#face-expression-detection)**
* **[Face Pose Estimation](#face-pose-estimation)**
* **[Eye Closeness Detection](#eye-closeness-detection)**
* **[Gender Detection](#gender-detection)**
* **[Age Detection](#age-detection)**
* **[Face Feature Embedding](#face-recognition)**

<h2>Examples</h2>

https://github.com/kby-ai/FaceRecognition-Javascript/assets/125717930/551b6964-0fef-4483-85a7-76792c0f3b56



* [Vue.js Demo](https://github.com/Faceplugin-ltd/FacePlugin-FaceRecognition-Vue)
* [React.js Demo](https://github.com/Faceplugin-ltd/FacePlugin-FaceRecognition-React)

[**Please subscribe on Youtube channel**]
<a href="http://www.youtube.com/watch?feature=player_embedded&v=1aogUPLjdtw" target="_blank">
 <img src="http://img.youtube.com/vi/1aogUPLjdtw/maxresdefault.jpg" alt="Watch the video" width="960" height="520" border="10" />
</a>


<h2>Documentation</h2>

Here are some useful documentation

<a name="face-detection"></a>
### Face Detection
Load detection model
```
loadDetectionModel()
```
Detect face in the image
```
detectFace(session, canvas_id)
```

<a name="face-landmark-extraction"></a>
### Face Landmark Extraction
Load landmark extraction model
```
loadLandmarkModel()
```
Extract face landmark in the image using detection result
```
predictLandmark(session, canvas_id, bbox)
```

<a name="face-liveness-detection"></a>
### Face Liveness Detection
Load liveness detection model
```
loadLivenessModel()
```
Detect face liveness in the image using detection result. (Anti-spoofing)
```
predictLiveness(session, canvas_id, bbox)
```

<a name="face-expression-detection"></a>
### Face Expression Detection
Load expression detection model
```
loadExpressionModel()
```
Detect face expression
```
predictExpression(session, canvas_id, bbox)
```

<a name="face-pose-estimation"></a>
### Face Pose Estimation
Load pose estimation model
```
loadPoseModel()
```
Predict facial pose
```
predictPose(session, canvas_id, bbox, question)
```

<a name="eye-closeness-detection"></a>
### Eye Closeness Detection
Load eye closeness model
```
loadEyeModel()
```
Predict eye closeness
```
predictEye(session, canvas_id, landmark)
```

<a name="gender-detection"></a>
### Gender Detection
Load gender detection model
```
loadGenderModel()
```
Predict gender using face image
```
predictGender(session, canvas_id, landmark)
```

<a name="age-detection"></a>
### Age Detection
Load age detection model
```
loadAgeModel()
```
Predict age using face image
```
predictAge(session, canvas_id, landmark)
```

<a name="face-recognition"></a>
### Face Recognition
Load feature extraction model
```
loadFeatureModel()
```
Extract face feature vector in 512 dimension
```
extractFeature(session, canvas_id, landmarks)
```

## Contact
If you want to get better model, please contact us

<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/faceplugin"><img src="https://img.shields.io/badge/telegram-@faceplugin-blue.svg?logo=telegram " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://wa.me/+14422295661"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp " alt="faceplugin.com">
</div>


