using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class BabyNeuralNetwork : SimpleNeuralNet, DisplayableNeuralNet
{
    System.Func<double, double> ActivationFunc;
    System.Func<double, double> ActivationDiffFunc;
    System.Func<double, double, double> ErrorFunc;
    System.Func<double, double, double> ErrorDiffFunc;

    public BabyNeuralNetwork(int[] layerSize,
        System.Func<double, double> activationFunc, System.Func<double, double> activationDiffFunc,
        System.Func<double, double, double> errorFunc, System.Func<double, double, double> errorDiffFunc) : base(layerSize)
    {
        ActivationFunc = activationFunc;
        ActivationDiffFunc = activationDiffFunc;
        ErrorFunc = errorFunc;
        ErrorDiffFunc = errorDiffFunc;
    }

    private Vector<double> ForwardProb(Vector<double> input)
    {
        return this.ForwardProb(input, ActivationFunc);
    }


    public Vector<double> Predict(Vector<double> input)
    {
        return ForwardProb(input);
    }

    public double[] Predict(double[] input)
    {
        Vector<double> vectorInput = Vector<double>.Build.Dense(input);
        return ForwardProb(vectorInput).ToArray();
    }

    public double TestErrorMag(System.Tuple<Vector<double>, Vector<double>>[] testSet)
    {
        double errorMagSum = 0.0;
        foreach (System.Tuple<Vector<double>, Vector<double>> t in testSet)
        {
            Vector<double> input = t.Item1;
            Vector<double> expectedOutput = t.Item2;
            Vector<double> output = Predict(input);

            Vector<double> errorV = LinearAlgebra.Apply(ErrorFunc, output, expectedOutput);
            errorMagSum += errorV.L2Norm();
        }

        return errorMagSum / testSet.Length;
    }

    public void BackProb(System.Tuple<Vector<double>, Vector<double>>[] inputExpectationSet)
    {
        this.BackProb(inputExpectationSet, ActivationFunc, ActivationDiffFunc, ErrorFunc, ErrorDiffFunc);
    }

    public void BackProb(System.Tuple<double[], double[]>[] inputExpectationSet)
    {
        BackProb(ConvertTupeSetArrayToVector(inputExpectationSet));
    }

    #region Train
    public void Train(System.Tuple<Vector<double>, Vector<double>>[] trainSet,
        System.Tuple<Vector<double>, Vector<double>>[] testSet, int epoch, System.Action<double> Callback)
    {
        int batch_size = (int)(trainSet.Length / epoch) > 0 ? (int)(trainSet.Length / epoch) : 1;

        // randomize input-expects set
        trainSet = random.Shuffle(trainSet);

        // train
        for (int i = 0; i < trainSet.Length; i += batch_size)
        {
            // get batch
            int batch_end_index = i + batch_size - 1 < trainSet.Length ? i + batch_size - 1 : trainSet.Length - 1;
            System.Tuple<Vector<double>, Vector<double>>[] batch = trainSet.ArrayFromIndexToInclude(i, batch_end_index);

            // train batch
            BackProb(batch);

            // error 
            double error = TestErrorMag(testSet);

            // Call back
            Callback?.Invoke(error);
        }
    }

    public IEnumerator IEmuneratorTrain(System.Tuple<Vector<double>, Vector<double>>[] trainSet,
        System.Tuple<Vector<double>, Vector<double>>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject)
    {
        int batch_size = (int)(trainSet.Length / epoch) > 0 ? (int)(trainSet.Length / epoch) : 1;

        // randomize input-expects set
        trainSet = random.Shuffle(trainSet);

        StartCallback?.Invoke(batch_size);

        double error = double.MaxValue;
        // train
        for (int i = 0; i < trainSet.Length; i += batch_size)
        {
            // get batch
            int batch_end_index = i + batch_size - 1 < trainSet.Length ? i + batch_size - 1 : trainSet.Length - 1;
            System.Tuple<Vector<double>, Vector<double>>[] batch = trainSet.ArrayFromIndexToInclude(i, batch_end_index);

            // train batch
            BackProb(batch);

            // error 
            error = TestErrorMag(testSet);

            // Call back
            UpdateCallback?.Invoke(error);
            yield return CallbackObject;
        }

        EndCallback?.Invoke(error);
    }

    public IEnumerator IEmuneratorTrain(System.Tuple<Vector<double>, Vector<double>>[] trainSet,
        System.Tuple<Vector<double>, Vector<double>>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<int,double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject, bool randomize = true)
    {
        int batch_size = (int)(trainSet.Length / epoch) > 0 ? (int)(trainSet.Length / epoch) : 1;

        // randomize input-expects set
        if (randomize)
            trainSet = random.Shuffle(trainSet);

        StartCallback?.Invoke(batch_size);

        double error = double.MaxValue;

        int poch = 0;
        // train
        for (int i = 0; i < trainSet.Length; i += batch_size)
        {
            // get batch
            int batch_end_index = i + batch_size - 1 < trainSet.Length ? i + batch_size - 1 : trainSet.Length - 1;
            System.Tuple<Vector<double>, Vector<double>>[] batch = trainSet.ArrayFromIndexToInclude(i, batch_end_index);

            // train batch
            BackProb(batch);

            // error 
            error = TestErrorMag(testSet);

            // Call back
            UpdateCallback?.Invoke(poch, error);

            poch++;
            yield return CallbackObject;
        }

        EndCallback?.Invoke(error);
    }

    // Array versions
    public void Train(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet, int epoch, System.Action<double> Callback)
    {
        System.Tuple<Vector<double>, Vector<double>>[] vectorTrainSet =   ConvertTupeSetArrayToVector(trainSet);
        System.Tuple<Vector<double>, Vector<double>>[] vectorTestSet =   ConvertTupeSetArrayToVector(testSet);

        Train(vectorTrainSet, vectorTestSet, epoch, Callback);
    }

    public IEnumerator IEmuneratorTrain(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<double> UpdateCallback, System.Action<double> EndCallback, 
        object CallbackObject)
    {
        System.Tuple<Vector<double>, Vector<double>>[] vectorTrainSet = ConvertTupeSetArrayToVector(trainSet);
        System.Tuple<Vector<double>, Vector<double>>[] vectorTestSet = ConvertTupeSetArrayToVector(testSet);

        return IEmuneratorTrain(vectorTrainSet, vectorTestSet, epoch, StartCallback, UpdateCallback, EndCallback, CallbackObject);
    }

    public IEnumerator IEmuneratorTrain(System.Tuple<double[], double[]>[] trainSet,
        System.Tuple<double[], double[]>[] testSet, int epoch,
        System.Action<int> StartCallback, System.Action<int, double> UpdateCallback, System.Action<double> EndCallback,
        object CallbackObject, bool randomize=true)
    {
        System.Tuple<Vector<double>, Vector<double>>[] vectorTrainSet = ConvertTupeSetArrayToVector(trainSet);
        System.Tuple<Vector<double>, Vector<double>>[] vectorTestSet = ConvertTupeSetArrayToVector(testSet);

        return IEmuneratorTrain(vectorTrainSet, vectorTestSet, epoch, StartCallback, UpdateCallback, EndCallback, CallbackObject);
    }

    #endregion


    #region Helper
    protected System.Tuple<Vector<T>,Vector<T>>[] ConvertTupeSetArrayToVector<T>(System.Tuple<T[], T[]>[] tupleSet) 
        where T:struct, System.IEquatable<T>, System.IFormattable
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
    #endregion
}
