var cv = null;

async function load_opencv() {
  if (!window.WebAssembly) {
    console.log("Your web browser doesn't support WebAssembly.")
    return
  }

  console.log("loading OpenCv.js")
  const script = document.createElement("script")
  script.type = "text/javascript"
  script.async = "async"
  script.src = "../js/opencv.js"
  document.body.appendChild(script)
  script.onload = () => {
    console.log("OpenCV.js is loaded.")
  }

  window.Module = {
    wasmBinaryFile: `../js/opencv_js.wasm`, // for wasm mode
    preRun: () => {
      console.log('preRun function on loading opencv')
    },
    _main: () => {
      console.log('OpenCV.js is ready.')
      cv = window.cv
    }
  }
  return
}

export {load_opencv, cv}