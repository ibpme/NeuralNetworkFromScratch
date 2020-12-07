window.addEventListener("load", () => {
  const canvas = document.querySelector("#canvas");
  const ctx = canvas.getContext("2d");
  canvas.height = 280;
  canvas.width = 280;
  ctx.fillStyle = "black";
  ctx.strokeStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  let touch = false;

  function startPosition(event) {
    touch = true;
    draw(event);
  }

  function finishedPosition() {
    touch = false;
    ctx.beginPath();
  }

  function draw(event) {
    if (!touch) return;
    ctx.lineWidth = 20;
    ctx.lineCap = "round";

    ctx.lineTo(event.clientX, event.clientY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(event.clientX, event.clientY);
  }

  canvas.addEventListener("mousedown", startPosition);
  canvas.addEventListener("mouseup", finishedPosition);
  canvas.addEventListener("mousemove", draw);

  function sigmoid(input) {
    if (input instanceof Array) {
      const recurse = input.map((x) => {
        return sigmoid(x);
      });

      return recurse;
    } else {
      return Number(1 / (1 + Math.exp(-Number(input))));
    }
  }

  function feedFoward(input, model) {
    let data = math.matrix(input);
    for (let layer = 0; layer < model.num_layers - 1; layer++) {
      const bias = math.matrix(model.biases[layer]);
      const weight = math.matrix(model.weights[layer]);
      data = sigmoid(math.add(math.multiply(weight, data), bias)._data);
    }
    return math.transpose(data);
  }

  const clearButton = document.getElementById("clear");

  clearButton.addEventListener("click", () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });

  const predictButton = document.getElementById("predict");
  console.log(predictButton);
  predictButton.addEventListener("click", async (e) => {
    e.preventDefault();

    async function preprocessImage() {
      const imageData = await IJS.Image.fromCanvas(canvas);
      const resizedImage = imageData.resize({ width: 28, height: 28 });
      const greyScaled = resizedImage.grey();
      return greyScaled.data;
    }
    const image = await preprocessImage();
    const input = Array.from(image);
    const output = feedFoward(math.matrix(input).resize([784, 1]), model)[0];
    console.log(output);

    result = output.indexOf(Math.max(...output));
    resultText = document.getElementById("result");
    resultText.innerText = "Result :" + result;
  });
});
