using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class SimpleNeuralNet
{
    protected System.Random random;
    protected Vector inputNodes;
    protected int inputSize;
    protected int[] layerSize;
    protected Vector<double>[] layerNodes;
    protected Matrix<double>[] weights;
    
    public SimpleNeuralNet(int[] layerSize)
    {
        random = new System.Random();

        this.layerSize = (int[])layerSize.Clone();

        inputSize = layerSize[0];
        InitializeRandomizedWeightsAndBias(layerSize, out layerNodes, out weights);
    }

    public void RandomizeWeightsAndBias()
    {
        InitializeRandomizedWeightsAndBias(this.layerSize, out this.layerNodes, out this.weights);
    }

    private void InitializeRandomizedWeightsAndBias(int[] layerSize, 
        out Vector<double>[] layerBias, out Matrix<double>[] weights)
    {
        layerBias = new Vector<double>[layerSize.Length - 1];
        weights = new Matrix<double>[layerSize.Length - 1];
        for (int i = 0; i < layerSize.Length - 1; i++)
        {
            layerBias[i] = LinearAlgebra.BuildRandomVector(layerSize[i + 1], -1, 1, random);
            weights[i] = LinearAlgebra.BuildRandomMatrix(layerSize[i + 1], layerSize[i], -1, 1, random);
        }
    }

    private void InitializeZerosWeightsAndBias(int[] layerSize,
        out Vector<double>[] layerBias, out Matrix<double>[] weights)
    {
        layerBias = new Vector<double>[layerSize.Length - 1];
        weights = new Matrix<double>[layerSize.Length - 1];
        for (int i = 0; i < layerSize.Length - 1; i++)
        {
            layerBias[i] = Vector<double>.Build.Dense(layerSize[i + 1], 0);
            weights[i] = Matrix<double>.Build.Dense(layerSize[i + 1], layerSize[i], 0);
        }
    }

    public Vector<double> ForwardProb(Vector<double> input, System.Func<double, double> ActivationFunc)
    {
        Vector<double> A = input.Clone();
        Vector<double> AO;
        for (int i=0; i<weights.Length; i++)
        {
            AO = LinearAlgebra.Apply(ActivationFunc, weights[i].Multiply(A).Add(layerNodes[i])); // activation(w_l * I + b_l)
            A = AO.Clone();
        }
        return A;
    }

    public Vector<double> ForwardProb(double[] input, System.Func<double, double> ActivationFunc)
    {
        return ForwardProb(Vector<double>.Build.Dense(input), ActivationFunc);
    }

    private System.Tuple<Vector<double>[], Vector<double>[]> ForwardProbActivation(Vector<double> input, System.Func<double, double> ActivationFunc)
    {
        Vector<double>[] activations = new Vector<double>[layerSize.Length - 1];
        Vector<double>[] linearizations = new Vector<double>[layerSize.Length - 1];

        Vector<double> A = input.Clone();
        Vector<double> AO;
        for (int i = 0; i < weights.Length; i++)
        {
            Vector<double> linearization = weights[i].Multiply(A).Add(layerNodes[i]);
            AO = LinearAlgebra.Apply(ActivationFunc, linearization); // activation(w_l * I + b_l)
            A = AO.Clone();
            activations[i] = AO;
            linearizations[i] = linearization;
        }
        return new System.Tuple<Vector<double>[], Vector<double>[]>(activations, linearizations);
    }

    // C = Error(a_l, expected)
    // z_l = w_l * a_(l-1) + b_l
    // a_l = ActivationFunc( z_l )
    // E_L = ErrorDifFunc(a_L, expected)
    // E_l = (w_[l+1])^T * E_[l+1]
    //
    // V_L = ActivationDifFunc(z_L) * E_L
    // V_l = (w_[l+1])^T * V_[l+1] (*) ActiDiff(z_l)
    //
    // dC/d(a_[l-1]) = w[l]^T * ActivationDifFunc(z_l) * E_l == w[l]^T * V_l
    // 
    // dC/d(w_[l]) = dz_[l]/dw_[l] * da_l/dz_l * dC/da_l
    //             == a_[l-1] * ActivationDifFunc(z_l) * E_l = a_[l-1] * V_l
    // dC/d(b_[l]) = 1 * ActivationDifFunc(z_l) * E_l = V_l
    public void BackProb(System.Tuple<Vector<double>, Vector<double>>[] inputExpectationSet,
        System.Func<double, double> ActivationFunc, System.Func<double,double> ActivationDifFunc,
        System.Func<double, double, double> ErrorFunc, System.Func<double,double,double> ErrorDifFunc)
    {
        Vector<double>[] sumDiffLayerBias = new Vector<double>[layerNodes.Length];
        Matrix<double>[] sumDiffWeights = new Matrix<double>[weights.Length];

        InitializeZerosWeightsAndBias(this.layerSize, out sumDiffLayerBias, out sumDiffWeights);

        int trainSetSize = inputExpectationSet.Length;

        for (int i=0; i< trainSetSize; i++)
        {
            System.Tuple<Vector<double>, Vector<double>> thisSet = inputExpectationSet[i];
            Vector<double> input = thisSet.Item1.Clone();
            Vector<double> expectedOutput = thisSet.Item2.Clone();

            System.Tuple<Vector<double>[],Vector<double>[]> forwardResult =  ForwardProbActivation(input, ActivationFunc);
            Vector<double>[] activations = forwardResult.Item1;
            Vector<double>[] linearizations = forwardResult.Item2;

            int activationLength = activations.Length;

            // back prob
            Vector<double>[] diffLayerBias;
            Matrix<double>[] diffWeights;
            InitializeZerosWeightsAndBias(this.layerSize, out diffLayerBias, out diffWeights);

            Vector<double> ActDiif_z_L = LinearAlgebra.Apply(ActivationDifFunc, linearizations[activationLength - 1]);
            Vector<double> ErrorDiff_a_L = LinearAlgebra.Apply(ErrorDifFunc, activations[activationLength - 1], expectedOutput);
            Vector<double> V =  ActDiif_z_L.PointwiseMultiply(ErrorDiff_a_L); // ActDiif_z_L (*) ErrorDiff_a_L

            for (int l = activationLength-1; l>0; l--)
            {
                // diff bias
                diffLayerBias[l] = V.Clone();

                // update by loop through weights_j
                for (int row=0; row<diffWeights[l].RowCount; row++)
                {
                    for (int col = 0; col < diffWeights[l].ColumnCount; col++)
                    {
                        diffWeights[l][row,col] = activations[l - 1][col] * V[row];
                    }
                }

                // update V_l -> V_[l-1]:  w_l^T * V_[l+1] (*) ActDiff(z_{l-1})
                V = weights[l].Transpose().Multiply(V).PointwiseMultiply(LinearAlgebra.Apply(ActivationDifFunc, linearizations[l-1]));
            }

            // last
            #region Test
            int lastl = 0;

            diffLayerBias[lastl] = V.Clone();

            // update by loop through weights_j
            for (int row = 0; row < diffWeights[lastl].RowCount; row++)
            {
                for (int col = 0; col < diffWeights[lastl].ColumnCount; col++)
                {
                    diffWeights[lastl][row, col] = input[col] * V[row];
                }
            }
            #endregion


            // update Diff sums
            for (int c=0; c<sumDiffLayerBias.Length; c++)
            {
                /*Debug.Log("Sum Layer: " + System.String.Join<Vector<double>>(", ", sumDiffLayerBias));  
                Debug.Log("Layer: " + System.String.Join<Vector<double>>(", ", diffLayerBias));
                Debug.Log("Sum Weights: " + System.String.Join<Matrix<double>>(", ", sumDiffWeights));
                Debug.Log("Weights: " + System.String.Join<Matrix<double>>(", ", diffWeights));*/

                sumDiffLayerBias[c] += diffLayerBias[c];
                sumDiffWeights[c] += diffWeights[c];
            }
        }

        // apply diff to weights and biases by avg diff sums
        for (int c = 0; c < sumDiffLayerBias.Length; c++)
        {
            layerNodes[c] -= sumDiffLayerBias[c] / trainSetSize;
            weights[c] -= sumDiffWeights[c] / trainSetSize;
        }
    }

    public void BackProb(System.Tuple<double[], double[]>[] inputExpectationSet,
        System.Func<double, double> ActivationFunc, System.Func<double, double> ActivationDifFunc,
        System.Func<double, double, double> ErrorFunc, System.Func<double, double, double> ErrorDifFunc)
    {
        System.Tuple<Vector<double>, Vector<double>>[] vectorTrainSet = new System.Tuple<Vector<double>, Vector<double>>[inputExpectationSet.Length];
        
        for (int i=0; i< inputExpectationSet.Length; i++)
        {
            System.Tuple<double[], double[]> t = inputExpectationSet[i];

            Vector<double> input = Vector<double>.Build.Dense(t.Item1);
            Vector<double> output = Vector<double>.Build.Dense(t.Item2);

            System.Tuple<Vector<double>, Vector<double>> vectorTuple = 
                new System.Tuple<Vector<double>, Vector<double>>(input, output);

            vectorTrainSet[i] = vectorTuple;
        }

        BackProb(vectorTrainSet, ActivationFunc, ActivationDifFunc, ErrorFunc, ErrorDifFunc);
    }

    #region Getters and Setters
    public int[] GetLayerSize()
    {
        return (int[]) layerSize.Clone();
    }

    public Matrix<double>[] GetWeights()
    {
        return (Matrix<double>[])weights.Clone();
    }

    public Vector<double>[] GetBiases()
    {
        return (Vector<double>[])layerNodes.Clone();
    }
    #endregion

    public Vector<double> GetErrorVectorFromInput(Vector<double> input, Vector<double> expected, 
        System.Func<Vector<double>, Vector<double>, Vector<double>> ErrorFunc)
    {
        if (expected.Count != layerSize[layerSize.Length-1])
        {
            throw new System.Exception("[Vector Size Not Mactch] Expected vector size does not match neural network output layer size: " +
                $"output layer size {layerSize[layerSize.Length - 1]}, but expected vector size {expected.Count}");
        }
        if (input.Count != layerSize[0])
        {
            throw new System.Exception("[Vector Size Not Mactch] Input vector size does not match neural network's input layer size: " +
                $"input layer size {layerSize[0]}, but input vector size {input.Count}");
        }

        Vector<double> output = ForwardProb(input, Math.Sigmoid);
        Vector<double> SEV = ErrorFunc(output, expected);

        return SEV;
    }

    public override string ToString()
    {
        string result = "";
        result += $"layer_I: Vector<{inputSize}>\n";
        for (int i=0; i< weights.Length; i++)
        {
            result += $"weight_matrix_{i+1}: Matrix<{weights[i].RowCount}x{weights[i].ColumnCount}> " +
                $"[max: {LinearAlgebra.GetMaxEntry(weights[i])}, min: {LinearAlgebra.GetMinEntry(weights[i])}]\n";
            result += $"layer_{i + 1}: Vector<{layerNodes[i].Count}> [max: {layerNodes[i].Maximum()}, min: {layerNodes[i].Minimum()}] \n";
        }

        return result;
    }
}
