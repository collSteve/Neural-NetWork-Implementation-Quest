using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Math
{
    public static double RandomDouble(double min, double max, System.Random random)
    {
        return random.NextDouble() * (max - min) + min;
    }

    public static double Sigmoid(double input)
    {
        return 1 / (1 + System.Math.Pow(System.Math.E, -input));
    }

    public static double SigmoidDiff(double input)
    {
        return Sigmoid(input) * (1 - Sigmoid(input));
    }

    public static double SigmoidNegative11(double input)
    {
        return (1 / (1 + System.Math.Pow(System.Math.E, -input)) - 0.5) * 2;
    }

    public static double SquareError(double num, double expected)
    {
        return (num - expected) * (num - expected);
    }

    public static double SquareErrorDiff(double num, double expected)
    {
        return 2 * (num - expected);
    }
}
