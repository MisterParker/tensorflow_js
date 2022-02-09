let net;

async function mlApp(){
    console.log('Loading the neural net (MobileNet in this case)...');

    // load the model
    net = await mobilenet.load()
    console.log('Succefully loaded model...');

    // load your image here & make predictions throght the above model
    const imgIn = document.getElementById('img');
    const result = await net.classify(imgIn);
    console.log(result);
}

mlApp();