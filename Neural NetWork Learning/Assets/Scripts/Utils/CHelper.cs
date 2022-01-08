using System;
using System.Collections;
using System.Collections.Generic;

public static class CHelper
{
    public static T[] SubArray<T>(this T[] data, int index, int length)
    {
        T[] result = new T[length];
        Array.Copy(data, index, result, 0, length);
        return result;
    }

    // [formIndex, toIndex]
    public static T[] ArrayFromIndexToInclude<T>(this T[] data, int fromIndex, int toIndex)
    {
        if (toIndex >= data.Length || toIndex < 0 || fromIndex >= data.Length || fromIndex < 0 )
        {
            throw new IndexOutOfRangeException("index out of range");
        }

        if (toIndex < fromIndex)
        {
            throw new IndexOutOfRangeException("FromIndex is expected to be smaller than toIndex. " +
                $"However, fromIndex [{fromIndex}] < toIndex [{toIndex}] are given.");
        }
        int length = toIndex - fromIndex + 1;
        T[] result = new T[length];
        Array.Copy(data, fromIndex, result, 0, length);
        return result;
    }

    // [fromIndex, toIndex)
    public static T[] ArrayFromIndexToExclude<T>(this T[] data, int fromIndex, int toIndex)
    {
        return ArrayFromIndexToInclude(data, fromIndex, toIndex - 1);
    }

    public static T[] Shuffle<T>(this Random rng, T[] array)
    {
        T[] copy = (T[]) array.Clone();
        int n = copy.Length;
        while (n > 1)
        {
            int k = rng.Next(n--);
            T temp = copy[n];
            copy[n] = copy[k];
            copy[k] = temp;
        }
        return copy;
    }

    public static T[] Shuffle<T>(this T[] array, Random rng)
    {
        return rng.Shuffle(array);
    }
}
