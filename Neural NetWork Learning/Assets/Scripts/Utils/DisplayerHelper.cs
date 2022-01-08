using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DisplayerHelper 
{
    public static Color ColorRulerRedGreen(float value, float min=-1, float max=1, float mid=0)
    {
        if (value >= mid)
        {
            float scale = (value - mid) / (max - mid);
            scale = Mathf.Clamp01(scale);

            return new Color(scale, 0f, 0f); // red
        }
        else
        {
            float scale = (mid-value) / ( mid-min);
            scale = Mathf.Clamp01(scale);
            return new Color(0f, scale, 0f); // green

        }
    }
}
