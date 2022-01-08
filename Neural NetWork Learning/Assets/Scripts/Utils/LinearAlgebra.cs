using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class LinearAlgebra
{
    public static Vector<double> BuildRandomVector(int numElems, double min, double max, System.Random random)
    {
        Vector<double> v = Vector<double>.Build.Dense(numElems);

        for (int i=0; i<v.Count; i++)
        {
            v[i] = Math.RandomDouble(min, max, random);
        }
        return v;
    }


    public static Matrix<double> BuildRandomMatrix(int numRows, int numCols, double min, double max, System.Random random)
    {
        Matrix<double> m = Matrix<double>.Build.Dense(numRows, numCols);

        // a(i,j) ith row, jth col
        for (int i=0; i<numRows; i++)
        {
            for (int j=0; j<numCols; j++)
            {
                m[i, j] = Math.RandomDouble(min, max, random);
            }
        }

        return m;
    }

    public static Vector<double> VectorSquareError(Vector<double> actual, Vector<double> expected)
    {
        if (actual.Count != expected.Count)
        {
            throw new System.Exception($"[VectorSquareError.Error] Unequal sized vectors: actual vector of size {actual.Count}, expected of size {expected.Count}");
        }

        Vector<double> result = Vector<double>.Build.Dense(actual.Count, 0);

        for (int i=0; i<actual.Count; i++)
        {
            result[i] = (actual[i] - expected[i]) * (actual[i] - expected[i]);
        }

        return result;
    }

    public static Vector<double> Apply(System.Func<double, double> func, Vector<double> v)
    {
        Vector<double> result = Vector<double>.Build.Dense(v.Count);
        for (int i=0; i<v.Count; i++)
        {
            result[i] = func(v[i]);
        }

        return result;
    }

    public static Vector<double> Apply(System.Func<double, double, double> func, Vector<double> v, double num)
    {
        Vector<double> result = Vector<double>.Build.Dense(v.Count);
        for (int i = 0; i < v.Count; i++)
        {
            result[i] = func(v[i], num);
        }

        return result;
    }

    public static Vector<double> Apply(System.Func<double, double, double> func, double num, Vector<double> v)
    {
        Vector<double> result = Vector<double>.Build.Dense(v.Count);
        for (int i = 0; i < v.Count; i++)
        {
            result[i] = func(num, v[i]);
        }

        return result;
    }

    public static Vector<double> Apply(System.Func<double, double, double> func, Vector<double> v1, Vector<double> v2)
    {
        Vector<double> result = Vector<double>.Build.Dense(v1.Count);
        for (int i = 0; i < v1.Count; i++)
        {
            result[i] = func(v1[i], v2[i]);
        }

        return result;
    }

    public static double GetMaxEntry(Matrix<double> m)
    {
        double max = m[0, 0];
        for (int i=0; i<m.RowCount; i++)
        {
            for (int j=0; j<m.ColumnCount; j++)
            {
                if (m[i,j] > max)
                {
                    max = m[i, j];
                }
            }
        }
        return max;
    }

    public static double GetMinEntry(Matrix<double> m)
    {
        double min = m[0, 0];
        for (int i = 0; i < m.RowCount; i++)
        {
            for (int j = 0; j < m.ColumnCount; j++)
            {
                if (m[i, j] < min)
                {
                    min = m[i, j];
                }
            }
        }
        return min;
    }
}
