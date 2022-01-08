using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class MathTest : MonoBehaviour
{
    System.Random random;
    // Start is called before the first frame update
    void Start()
    {
        random = new System.Random();

        Vector<double> v = LinearAlgebra.BuildRandomVector(10, -1, 1, random);
        Matrix<double> m = LinearAlgebra.BuildRandomMatrix(10, 10, -1, 1, random);

        Vector<double> v1 = Vector<double>.Build.Dense(new double[] { 1, 2 });
        Vector<double> v2 = Vector<double>.Build.Dense(new double[] { 2, 3 });
        /*        Debug.Log($"random vector: {v}");
                Debug.Log($"random matrix: {m}");*/

        Debug.Log(v1.PointwiseMultiply(v2));
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
