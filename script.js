// Step 1 of 7 — Prepare Data for Exploratory Data Analysis
import { TRAINING_DATA as trainingData } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// Step 2 of 7 — Visualize and Explore the Data
// TODO Show some example images
// TODO show chart for example input and histogram of number images

// Step 3 of 7 — Prepare the Data for Machine Learning
tf.util.shuffleCombo(trainingData.inputs, trainingData.outputs);
console.log(trainingData.inputs[0])
let tensorIn = tf.tensor2d(trainingData.inputs);
let tensorOut = tf.oneHot(tf.tensor1d(trainingData.outputs, "int32"), 10);

// Step 4 of 7 — Build Machine Learning Model
let model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [784], units: 64, activation: "relu" }),
    tf.layers.dense({ units: 32, activation: "relu" }),
    tf.layers.dense({ units: 16, activation: "relu" }),
    tf.layers.dense({ units: 10, activation: "softmax" }),
  ],
});
model.compile({
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
  optimizer: "adam",
});

// Step 5 of 7 — Train the Model
let bestWeights;
let lowestLoss = Number.MAX_SAFE_INTEGER;
let history = train();

// Step 6 of 7 — Evaluate the Model Performance
history = history.then(function (data) {
  console.log(data.history);
  // TODO show a chart of the loss per epoch
  // TODO show a chart of the accuracy per epoch
});

// Step 7 of 7 — Enable Ad-Hoc Verification of the Model
history.then(function () {
  evaluate();
  function evaluate() {
    let OFFSET = Math.floor(Math.random() * trainingData.inputs.length);
    let answer = tf.tidy(function () {
      let newInput = tf.tensor1d(trainingData.inputs[OFFSET]).expandDims();
      let output = model.predict(newInput);

      return output.squeeze().argMax();
    });

    answer.array().then(function (index) {
      PREDICTION_ELEMENT.innerText = index;
      PREDICTION_ELEMENT.setAttribute(
        "class",
        index === trainingData.outputs[OFFSET] ? "correct" : "wrong"
      );
      answer.dispose();
      drawImage(trainingData.inputs[OFFSET]);
    });
  }
  function drawImage(digit) {
    let imageData = CTX.getImageData(0, 0, 28, 28);
    console.log(digit);
    digit.forEach((v, i) => {
      imageData.data[i * 4 + 0] = v * 255; // R
      imageData.data[i * 4 + 1] = v * 255; // G
      imageData.data[i * 4 + 2] = v * 255; // B
      imageData.data[i * 4 + 3] = 1 * 255; // Alpha
    });
    CTX.putImageData(imageData, 0, 0);
    setTimeout(evaluate, 6000); //TODO change to button
  }
});
// Helping functions
async function train() {
  // Trains the model. Note: we have configured it to end when learning appears done, and then it restores the best weights
  history = await model.fit(tensorIn, tensorOut, {
    batchSize: 1024,
    callbacks: [
      new tf.CustomCallback({ onEpochEnd: saveBestWeights }),
      tf.callbacks.earlyStopping({
        patience: 10,
      }),
    ],
    epochs: 100,
    shuffle: true,
    validationSplit: 0.15,
  });
  model.loadWeights(bestWeights, false);
  tensorIn.dispose();
  tensorOut.dispose();
  return history;
}

function saveBestWeights(batch, logs) {
  if (logs.val_loss < lowestLoss) {
    lowestLoss = logs.val_loss;
    bestWeights = model.weights;
  }
}
const PREDICTION_ELEMENT = document.getElementById("prediction");
const PREDICTION_ELEMENT2 = document.getElementById("prediction2");
const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d", { willReadFrequently: true });
const draw = (e) => {
  if (isPainting) {
    ctxDrawing.lineWidth = 10;
    ctxDrawing.lineCap = "round";
    ctxDrawing.lineTo(e.clientX - canvasDrawing.offsetLeft, e.clientY - canvasDrawing.offsetTop);
    ctxDrawing.stroke();
  }
};
function rgbToGrayscale(red, green, blue) {
    const r = red * .30;
    const g = green * .59;
    const b = blue * .11;
    const gray = r + g + b;
    return gray
}


/*Enable Drawing and validation*/
let canvasDrawing = document.getElementById("drawing");
let ctxDrawing = canvasDrawing.getContext("2d");
let isPainting = false;
let startX;
let startY;

canvasDrawing.width = 100;
canvasDrawing.height = 100;


canvasDrawing.addEventListener("mousedown", (e) => {
  isPainting = true;
  startX = e.clientX;
  startY = e.clientY;
  ctxDrawing.stroke();
  ctxDrawing.beginPath();
});
canvasDrawing.addEventListener("mouseup", (e) => {
  isPainting = false;
  ctxDrawing.stroke();
  ctxDrawing.beginPath();
});
canvasDrawing.addEventListener ("mouseout", (e) => {
  isPainting = false;
  ctxDrawing.stroke();
  ctxDrawing.beginPath();
});


canvasDrawing.addEventListener("mousemove", draw);

document.getElementById("btnClear").addEventListener("click", function(e){
  e.preventDefault();
  ctxDrawing.clearRect(0, 0, canvasDrawing.width, canvasDrawing.height);
});
document.getElementById("btnSubmit").addEventListener("click", function(e){
  e.preventDefault();
  let img = getCanvasImage(canvasDrawing);
  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(img).expandDims();
    let output = model.predict(newInput);
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    PREDICTION_ELEMENT2.innerText = index;
    answer.dispose();
  });
  let imageData = CTX.getImageData(0, 0, 28, 28);
  img.forEach((v, i) => {
    imageData.data[i * 4 + 0] = v * 255; // R
    imageData.data[i * 4 + 1] = v * 255; // G
    imageData.data[i * 4 + 2] = v * 255; // B
    imageData.data[i * 4 + 3] = 1 * 255; // Alpha
  });
  CTX.putImageData(imageData, 0, 0);
});

function getCanvasImage(c){
  let c1 = document.createElement("canvas");
  let ctx1 = c1.getContext('2d')
  c1.width = 28
  c1.height = 28
  ctx1.drawImage(c, 4, 4, 20, 20);
  var imgData = ctx1.getImageData(0, 0, 28, 28);
  var arr = []
  for (var i = 0; i < imgData.data.length; i += 4) {
    arr.push(
      rgbToGrayscale(
        imgData.data[i+1],
        imgData.data[i+2],
        imgData.data[i+3]
      )
    )
  }
  return arr
}

