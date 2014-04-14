package faster_pca;


import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * A slimmer version of the center filter, centers data onto the mean
 * @author toddbodnar
 */
public class fast_center {
    /*public fast_center(double vals[][])
    {
        
    }*/
    
    /**
     * Normalize by centering on mean
     * @param means 
     */
    public fast_center(double means[])
    {
        this.means = means;
        vars = new double[means.length];
        
        for(int ct=0;ct<vars.length;ct++)
            vars[ct] = 1;
    }
    
    /**
     * Normalize by centering on mean, change variance to 1
     * @param means
     * @param stdevs 
     */
    public fast_center(double means[], double vars[])
    {
        this.means = means;
        this.vars = vars;
    }
    
    
    public Instance filter(Instance toFilter)
    {
        double weights[] = toFilter.toDoubleArray();
            for(int j=0;j<means.length;j++)
            {
                weights[j] = (weights[j] - means[j])/vars[j];
            }
        Instance inst = new DenseInstance(toFilter.weight(), weights);
        inst.setDataset(toFilter.dataset());
        return inst;
    }
    
    double means[], vars[];
}
