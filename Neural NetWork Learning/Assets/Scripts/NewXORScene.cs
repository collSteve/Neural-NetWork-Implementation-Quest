using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


public class NewXORScene : MonoBehaviour
{

    private DisplayableNeuralNet neuralNet;

    private Dictionary<Tuple<int,int>, GameObject> nodeGOs; // (layer index, node index) -> o  i.e. (2,1) second node in third layer
    // ([(l1, n1), (l2, n2)]) -> o : returns weight objects represent the weight from n1 node in l1 layer to n2 node in l2 layer
    private Dictionary<Tuple<Tuple<int,int>, Tuple<int, int>>, GameObject> weightGOs;

    [SerializeField]
    private GameObject nodeObject;

    private Func<float, Color> ColorRuler;
    private Coroutine trainCoroutine;
    private bool trainCoroutineRunning = false;

    // displays UI
    Texture2D XORTexture;
    int XORTextureSize;
    public RawImage XORImage;
    public Text percentageTextUI;

    // Start is called before the first frame update
    void Start()
    {
        // nodeObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);

        nodeObject.SetActive(false);

        ColorRuler = (v)=> DisplayerHelper.ColorRulerRedGreen(v);

        // Initialize XOR pic
        XORTextureSize = 100;
        XORTexture = new Texture2D(XORTextureSize, XORTextureSize);
        XORImage.texture = XORTexture;
        percentageTextUI.text = "NaN";

        int[] sampleLayerSize = new int[] { 2, 4, 1 };
        neuralNet = new BabyNeuralNetwork(sampleLayerSize, Math.Sigmoid, Math.SigmoidDiff, Math.SquareError, Math.SquareErrorDiff);


        InitialDisplayNeuralNet(neuralNet, new Vector2(-8f, 2.5f), new Vector2(4, -4f), 1, out nodeGOs, out weightGOs, nodeObject, ColorRuler);

        Tuple<double[], double[]>[] sampleTrainingSetBase = new Tuple<double[], double[]>[] {
            new Tuple<double[], double[]>(new double[2]{0, 0}, new double[1]{ 1}),
            new Tuple<double[], double[]>(new double[2]{1, 0}, new double[1]{ 0}),
            new Tuple<double[], double[]>(new double[2]{0, 1}, new double[1]{ 0}),
            new Tuple<double[], double[]>(new double[2]{1, 1}, new double[1]{ 1}),
        };

        // construct train set
        int times = 10000;
        Tuple<double[], double[]>[] sampleTrainingSet = new Tuple<double[], double[]>[times * sampleTrainingSetBase.Length];
        for (int i=0; i<sampleTrainingSet.Length; i+=sampleTrainingSetBase.Length)
        {
            for (int j = 0; j < sampleTrainingSetBase.Length; j++)
            {
                sampleTrainingSet[i + j] = sampleTrainingSetBase[j];
            }
        }


        System.Tuple<double[], double[]>[] sampleTestSet = new System.Tuple<double[], double[]>[] {
            new System.Tuple<double[], double[]>(new double[2]{0, 0}, new double[1]{ 1}),
            new System.Tuple<double[], double[]>(new double[2]{1, 0}, new double[1]{ 0}),
            new System.Tuple<double[], double[]>(new double[2]{0, 1}, new double[1]{ 0}),
            new System.Tuple<double[], double[]>(new double[2]{1, 1}, new double[1]{ 1}),
        };

        trainCoroutine = StartCoroutine(TrainNeuralNet(sampleTrainingSet, sampleTestSet, times, waitSeconds: 0f, false));

        // BackProbTest(neuralNet, sampleTrainingSetBase, 3000);
    }

    void InitialDisplayNeuralNet(DisplayableNeuralNet net, 
        Vector2 upleftCorner, Vector2 lowrightCorner, float nodeWidth, 
        out Dictionary<Tuple<int, int>, GameObject> nodeGOs, 
        out Dictionary<Tuple<Tuple<int, int>, Tuple<int, int>>, GameObject> weightGOs, 
        GameObject nodeObject, Func<float, Color> ColorRuler,
        float zPos=0)
    {
        int[] layerSize = net.GetLayerSize();
        Matrix<double>[] weights = net.GetWeights();
        Vector<double>[] biases = net.GetBiases();

        // initialize outputs
        nodeGOs = new Dictionary<Tuple<int, int>, GameObject>();
        weightGOs = new Dictionary<Tuple<Tuple<int, int>, Tuple<int, int>>, GameObject>();

        // - x -- x -- x -  : layerGapX: --
        float layerGapX = ((lowrightCorner.x - upleftCorner.x)) / layerSize.Length;

        // position nodes
        for (int i=0; i < layerSize.Length; i++)
        {
            // x position of layer
            float x = upleftCorner.x + layerGapX / 2 + layerGapX * i;

            // y position for nodes in layer
            float nodeGapY = ((upleftCorner.y - lowrightCorner.y)) / layerSize[i];
            for (int j=0; j < layerSize[i]; j++)
            {
                float y = lowrightCorner.y + nodeGapY / 2 + nodeGapY * j;

                // draw node at (x,y)
                GameObject o = Instantiate(nodeObject);
                o.transform.position = new Vector3(x, y, zPos);
                o.transform.localScale = new Vector3(nodeWidth, nodeWidth, nodeWidth);
                o.SetActive(true);

                nodeGOs.Add(new Tuple<int, int>(i, j), o); // j node in i layer 
            }
        }

        // position weights
        for (int k=1; k < layerSize.Length; k++)
        {
            int prevLayerIndex = k - 1;
            int layerIndex = k;

            Matrix<double> weightsLayer = weights[prevLayerIndex];

            // weightsLayer[i,j] == weight from j to i in forward prob
            for (int i = 0; i < layerSize[layerIndex]; i++)
            {
                for (int j = 0; j < layerSize[prevLayerIndex]; j++)
                {
                    // (l1=prevLayerIndex, n1=j) -> (l2=layerIndex, n2=i)
                    double weight = weightsLayer[i, j];

                    Tuple<int, int> fromNodeIndex = new Tuple<int, int>(prevLayerIndex, j);
                    Tuple<int, int> toNodeIndex = new Tuple<int, int>(layerIndex, i);

                    GameObject fromNode = nodeGOs[fromNodeIndex];
                    GameObject toNode = nodeGOs[toNodeIndex];

                    Color c = ColorRuler((float)weight);
                    GameObject weightO = UnityHelper.DrawLine(fromNode.transform.position, toNode.transform.position, c, lineWidth: 0.1f);
                    weightO.name = $"weight line [{fromNodeIndex.Item1}, {fromNodeIndex.Item2}]->[{toNodeIndex.Item1}, {toNodeIndex.Item2}]";

                    weightGOs.Add(new Tuple<Tuple<int, int>, Tuple<int, int>>(fromNodeIndex, toNodeIndex), weightO);
                }
            }
        }
    }

    void UpdateNodes(DisplayableNeuralNet net, Func<float, Color> ColorRuler)
    {
        int[] layerSize = net.GetLayerSize();
        Vector<double>[] biases = net.GetBiases();

        for (int i = 0; i < layerSize.Length; i++)
        {
            Vector<double> layerBiases;
            if (i ==0)
            {
                layerBiases = Vector<double>.Build.Dense(layerSize[0], 0);
            } 
            else
            {
                layerBiases = biases[i - 1];
            }
            for (int j = 0; j < layerSize[i]; j++)
            {
                double bias = layerBiases[j];
                Tuple<int, int> nodeIndex = new Tuple<int, int>(i, j);
                GameObject nodeO = nodeGOs[nodeIndex];

                nodeO.GetComponent<Renderer>().material.color = ColorRuler((float)bias);
            }
        }
    }

    void UpdateWeights(DisplayableNeuralNet net, Func<float, Color> ColorRuler)
    {
        int[] layerSize = net.GetLayerSize();
        Matrix<double>[] weights = net.GetWeights();

        for (int k = 1; k < layerSize.Length; k++)
        {
            int prevLayerIndex = k - 1;
            int layerIndex = k;

            Matrix<double> weightsLayer = weights[prevLayerIndex];

            // weightsLayer[i,j] == weight from j to i in forward prob
            for (int i = 0; i < layerSize[layerIndex]; i++)
            {
                for (int j = 0; j < layerSize[prevLayerIndex]; j++)
                {
                    // (l1=prevLayerIndex, n1=j) -> (l2=layerIndex, n2=i)
                    double weight = weightsLayer[i, j];

                    Tuple<int, int> fromNodeIndex = new Tuple<int, int>(prevLayerIndex, j);
                    Tuple<int, int> toNodeIndex = new Tuple<int, int>(layerIndex, i);

                    GameObject fromNode = nodeGOs[fromNodeIndex];
                    GameObject toNode = nodeGOs[toNodeIndex];

                    GameObject weightO = weightGOs[new Tuple<Tuple<int, int>, Tuple<int, int>>(fromNodeIndex, toNodeIndex)];

                    LineRenderer lr = weightO.GetComponent<LineRenderer>();
                    Color c = ColorRuler((float)weight);
                    lr.startColor = c;
                    lr.endColor = c;
                }
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (trainCoroutineRunning)
        {
            DisplayXORPic(neuralNet, XORTexture, XORTextureSize);
        }
    }

    IEnumerator TrainNeuralNet(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet,
        int epoch, float waitSeconds = 0.001f, bool randomize = true)
    {
        trainCoroutineRunning = true;

        System.Action<int> StartCallbackFunc = (batch_size) => {
            Debug.Log($"Training Start. Batch size: {batch_size}");
        };

        System.Action<int, double> UpdateCallbackFunc = (numEpoch, error) => {
            // Debug.Log($"Epoch [{numEpoch}]  |  Error: {error}");
            percentageTextUI.text = $"{(double)(numEpoch+1)/epoch * 100}% ({numEpoch}/{epoch})"; 

            // update new neural net display
            UpdateWeights(neuralNet, ColorRuler);
            UpdateNodes(neuralNet, ColorRuler);
            // DisplayXORPic(neuralNet, XORTexture, XORTextureSize);
        };

        System.Action<double> EndCallbackFunc = (error) => {
            Debug.Log($"Training Ends. Final error: {error}");
            trainCoroutineRunning = false;
        };

        Debug.Log($"Training Set Length: {trainSet.Length}");
        IEnumerator coroutine = neuralNet.IEmuneratorTrain(trainSet, testSet, epoch, 
            StartCallbackFunc, UpdateCallbackFunc, EndCallbackFunc, new WaitForSeconds(waitSeconds), randomize:randomize);

        
        return coroutine;
    }

    public void DisplayXORPic(DisplayableNeuralNet net, Texture2D XORTexture, int XORTextureSize)
    {

        for (int x = 0; x < XORTextureSize; x++)
        {
            for (int y = 0; y < XORTextureSize; y++)
            {
                float result = (float)net.Predict(new double[] { (double)x / (XORTextureSize), (double)y / (XORTextureSize) })[0];
                XORTexture.SetPixel(x, y, new Color(result, result, result));
            }
        }

        XORTexture.Apply();

    }

    void BackProbTest(DisplayableNeuralNet net, Tuple<double[], double[]>[] trainset, int times=1000)
    {
        for (int i=0; i<times; i++)
        {
            net.BackProb(trainset);
        }
        DisplayXORPic(net, XORTexture, XORTextureSize);
        Debug.Log("Finished Test");
    }

    private void OnDrawGizmos()
    {
    }

    #region Helpers
    #endregion
}
