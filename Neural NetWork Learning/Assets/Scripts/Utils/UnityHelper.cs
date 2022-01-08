using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityHelper
{
    public static GameObject DrawLine(Vector3 start, Vector3 end, Color color, float lineWidth = 0.1f)
    {
        GameObject myLine = new GameObject();
        myLine.transform.position = start;
        myLine.AddComponent<LineRenderer>();
        LineRenderer lr = myLine.GetComponent<LineRenderer>();
        lr.material = new Material(Shader.Find("Sprites/Default"));

        lr.positionCount = 2;

        lr.startColor = color;
        lr.endColor = color;

        lr.startWidth = lineWidth;
        lr.endWidth = lineWidth;

        Vector3[] linePositions = new Vector3[] { start, end };

        lr.SetPositions(linePositions);

        return myLine;
    }
}
