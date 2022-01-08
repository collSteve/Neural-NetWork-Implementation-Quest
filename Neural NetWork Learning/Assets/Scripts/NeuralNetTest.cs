using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetTest : MonoBehaviour
{
    SimpleNeuralNet neuralNet;

    public InputField[] Inputs;
    public InputField backProbInput;
    public RawImage display;
    public Texture2D displayTexture;

    private int textureSize = 100;

    double[] sampleInput;

    System.Tuple<double[], double[]>[] sampleTrainingSet;
    // Start is called before the first frame update
    void Start()
    {
        neuralNet = new SimpleNeuralNet(new int[] { 2, 3, 1 });

        displayTexture = new Texture2D(textureSize, textureSize);
        display.texture = displayTexture;

        Debug.Log("Net: " + neuralNet.ToString());

        // add gate
        sampleTrainingSet = new System.Tuple<double[], double[]>[] { 
            new System.Tuple<double[], double[]>(new double[2]{0, 0}, new double[1]{ 1}),
            new System.Tuple<double[], double[]>(new double[2]{1, 0}, new double[1]{ 0}),
            new System.Tuple<double[], double[]>(new double[2]{0, 1}, new double[1]{ 0}),
            new System.Tuple<double[], double[]>(new double[2]{1, 1}, new double[1]{ 1}),
        };

        sampleInput = new double[2] { 1, 1 };

        DisplayPic();
    }

    private void SetInput()
    {
        for (int i=0; i<Inputs.Length; i++)
        {
            InputField inputField = Inputs[i];
            double input = System.Convert.ToDouble(inputField.text);
            
            if (i<sampleInput.Length)
            {
                sampleInput[i] = input;
            }
        }
    }

    public void PassInput()
    {
        SetInput();

        Vector<double> sampleInputs = Vector<double>.Build.DenseOfArray(sampleInput);
        Vector<double> output = neuralNet.ForwardProb(sampleInputs, Math.Sigmoid);
        Debug.Log("output: " + output);

        // draw point
        displayTexture.SetPixel( sampleInput[0] < 1 ? (int)(sampleInput[0]*textureSize): textureSize-1, sampleInput[1] < 1 ? (int)(sampleInput[1]*textureSize) : textureSize - 1, Color.red);
        displayTexture.Apply();
    }

    public void RandomizeWeightsAndBias()
    {
        neuralNet.RandomizeWeightsAndBias();
        DisplayPic();
    }

    public void PrintNeuralNet()
    {
        Debug.Log("Net: " + neuralNet.ToString());
    }

    public void BackProb()
    {
        int numProbs = System.Convert.ToInt32(backProbInput.text);

        Debug.Log($"Back Prob of size {numProbs} starts");
        for (int i=0; i<numProbs; i++)
        {
            neuralNet.BackProb(sampleTrainingSet, Math.Sigmoid, Math.SigmoidDiff, Math.SquareError, Math.SquareErrorDiff);
        }

        Debug.Log($"Back Prob finished");

        DisplayPic();
    }

    public void DisplayPic()
    {

        for (int x = 0; x < textureSize; x++)
        {
            for (int y = 0; y < textureSize; y++)
            {
                float result = (float) neuralNet.ForwardProb(new double[] { (double)x/(textureSize), (double)y /(textureSize) }, Math.Sigmoid)[0];
                displayTexture.SetPixel(x, y, new Color(result, result, result));
            }
        }

        displayTexture.Apply();
        
    }


    // Update is called once per frame
    void Update()
    {
    }
}
