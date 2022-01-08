using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public interface DisplayableNeuralNet
{
    public IEnumerator IEmuneratorTrain(System.Tuple<Vector<double>, Vector<double>>[] trainSet,
        System.Tuple<Vector<double>, Vector<double>>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<int, double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject, bool randomize = true);

    public IEnumerator IEmuneratorTrain(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<int, double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject, bool randomize=true);

    public int[] GetLayerSize();

    public Matrix<double>[] GetWeights();

    public Vector<double>[] GetBiases();

    public double[] Predict(double[] input);

    public void BackProb(System.Tuple<Vector<double>, Vector<double>>[] inputExpectationSet);

    public void BackProb(System.Tuple<double[], double[]>[] inputExpectationSet);

    /*public IEnumerator IEmuneratorTrain(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<int, double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject)
    {
        System.Tuple<Vector<double>, Vector<double>>[] vectorTrainSet = ConvertTupeSetArrayToVector(trainSet);
        System.Tuple<Vector<double>, Vector<double>>[] vectorTestSet = ConvertTupeSetArrayToVector(testSet);

        return IEmuneratorTrain(vectorTrainSet, vectorTestSet, epoch, StartCallback, UpdateCallback, EndCallback, CallbackObject);
    }

    #region Helper
    private System.Tuple<Vector<T>, Vector<T>>[] ConvertTupeSetArrayToVector<T>(System.Tuple<T[], T[]>[] tupleSet)
        where T : struct, System.IEquatable<T>, System.IFormattable
    {
        System.Tuple<Vector<T>, Vector<T>>[] vectorSet = new System.Tuple<Vector<T>, Vector<T>>[tupleSet.Length];

        for (int i = 0; i < tupleSet.Length; i++)
        {
            System.Tuple<T[], T[]> t = tupleSet[i];

            Vector<T> input = Vector<T>.Build.Dense(t.Item1);
            Vector<T> output = Vector<T>.Build.Dense(t.Item2);

            System.Tuple<Vector<T>, Vector<T>> vectorTuple =
                new System.Tuple<Vector<T>, Vector<T>>(input, output);

            vectorSet[i] = vectorTuple;
        }
        return vectorSet;
    }
    #endregion*/
}
