let net;
const webCamElement = document.getElementById('webcam');

async function mlApp(){
    console.log('Loading the neural net (MobileNet in this case)...');

    // load the model
    net = await mobilenet.load()
    console.log('Succefully loaded model...');

    console.log('classifying the static image....')
    // load your image here & make predictions throght the above model
    const imgIn = document.getElementById('img');
    const result_img = await net.classify(imgIn);
    console.log(result_img);

    console.log('classifying the video feed....')
    const webCam = await tf.data.webcam(webCamElement);
    while(true) {
        const imgVid = await webCam.capture();
        const result_vid = await net.classify(imgVid);
        
        document.getElementById('console').innerText = `
            prediction: ${result_vid[0].className}\n
            probability: ${result_vid[0].probability}
        `;

        // dispose the tensor to release the memory
        imgVid.dispose();

        // add buffer before capturing the next frame
        await tf.nextFrame();

    }
}

mlApp();