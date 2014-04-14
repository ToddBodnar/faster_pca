package faster_pca;


import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * A slimmer version of the normalize filter, which does a linear transform of the data into [-1,1]
 * @author toddbodnar
 */
public class fast_normalize {
    /*public fast_center(double vals[][])
    {
        
    }*/
    
    
    /**
     * Normalize by changing range to 0-1
     * @param means
     * @param stdevs 
     */
    public fast_normalize(double min[], double max[])
    {
        this.min = min;
        this.max = max;
    }
    
    
    public Instance filter(Instance toFilter)
    {
        double weights[] = toFilter.toDoubleArray();
            for(int j=0;j<min.length;j++)
            {
                weights[j] = (weights[j] - min[j])/(max[j] - min[j])*2 - 1;
            }
        Instance inst = new DenseInstance(toFilter.weight(), weights);
        inst.setDataset(toFilter.dataset());
        return inst;
    }
    
    double min[], max[];
}
