let net;
const webCamElement = document.getElementById('webcam');
const classifier = knnClassifier.create(); // create a new classifier

async function mlApp(){
    console.log('Loading the neural net (MobileNet in this case)...');
    
    // load the model
    net = await mobilenet.load();
    console.log('Succefully loaded model...');

    // create an object which will capture images for Tensorflow API using the web camera as a tensor
    const webcam = await tf.data.webcam(webCamElement);

    // read the image from the webcam & associate a class to it
    const addExample = async classId => {
        // capture the image
        const img = await webcam.capture();

        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(img, true);

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);

        // dispose the tensor to release the memory
        img.dispose();
    };

    // add example for a particular class on clicking the button
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-b').addEventListener('click', () => addExample(2));

    while (true){
        if (classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // get the activation from your neural net from the webcam
            const activation = net.infer(img, 'conv_preds');
            // get the most likely class & the confidence ration from the classifier
            const result = await classifier.predictClass(activation);
            
            const classes =['A', 'B', 'C'];
            document.getElementById('console').innerText = `
                prediction: ${classes[result.label]}\n
                probability: ${result.confidences[result.label]}
            `;

            // dispose the tensor to release the memory
            img.dispose();
        }
        await tf.nextFrame();
    }
}

mlApp();


// code till webcam capture
// async function mlApp(){
//     console.log('Loading the neural net (MobileNet in this case)...');

//     // load the model
//     net = await mobilenet.load()
//     console.log('Succefully loaded model...');

//     console.log('classifying the static image....')
//     // load your image here & make predictions throght the above model
//     const imgIn = document.getElementById('img');
//     const result_img = await net.classify(imgIn);
//     console.log(result_img);

//     console.log('classifying the video feed....')
//     // create an object which will capture images for Tensorflow API using the web camera as a tensor
//     const webCam = await tf.data.webcam(webCamElement);
//     while(true) {
//         const imgVid = await webCam.capture();
//         const result_vid = await net.classify(imgVid);
        
//         document.getElementById('console').innerText = `
//             prediction: ${result_vid[0].className}\n
//             probability: ${result_vid[0].probability}
//         `;

//         // dispose the tensor to release the memory
//         imgVid.dispose();

//         // add buffer before capturing the next frame
//         await tf.nextFrame();

//     }
// }

// mlApp();