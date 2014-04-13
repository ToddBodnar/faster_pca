
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;
import weka.datagenerators.classifiers.classification.RDG1;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Testing of faster_pca.java
 * @author toddbodnar
 */
public class test {
    public static final int number_of_tests = 24;
    public static int passed = 0;
    
    /**
     * Times the execution of PCA
     * @param numAttributes the number of attributes in the generated dataset
     * @param numExamples the number of examples in the generated dataset
     * @param center center the data?
     * @param fast use faster_pca?
     * @return
     * @throws Exception 
     */
    public static long testSpeed(int numAttributes, int numExamples, boolean center, boolean fast) throws Exception
    {
        RDG1 randomdatagen = new RDG1();
        randomdatagen.setNumAttributes(numAttributes);
        randomdatagen.setNumExamples(numExamples);
        randomdatagen.defineDataFormat();
        Instances data = randomdatagen.generateExamples();
        
        PrincipalComponents pca = fast?(new faster_pca()):(new PrincipalComponents());
        pca.setCenterData(center);
            
        long time = System.currentTimeMillis();
        pca.setInputFormat(data);
            
        Filter.useFilter(data, pca);
        return System.currentTimeMillis() - time;
    }
    
    public static void testSpeed()
    {
        double time_sensitivity = .75; //how much lower we have to go to pass
        System.out.println("In speed tests ("+((1-time_sensitivity)*100)+"% improvement required to pass)");
        try
        {
            System.out.print("Comparing to wekaPCA (no center, 100x10000). weka takes: ");
            
            long weka_time = testSpeed(100,10000,false,false);
            System.out.print(weka_time+" fast pca takes: ");
            
            
            long fpca_time = testSpeed(100,10000,false,true);
            
            System.out.print(fpca_time+"\t");
            if(fpca_time < weka_time*time_sensitivity)
            {
                System.out.println("PASS");
                passed++;
            }
            else
            {
                System.out.println("FAIL");
            }
            
        }catch(Exception ex)
        {
            System.out.println("FAIL");
            Logger.getLogger(test.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        
        try
        {
            System.out.print("Comparing to wekaPCA (center, 100x10000). weka takes: ");
            
            
            long weka_time = testSpeed(100,10000,true,false);
            System.out.print(weka_time+" fast pca takes: ");
            
            
            long fpca_time = testSpeed(100,10000,true,true);
            
            System.out.print(fpca_time+"\t");
            if(fpca_time < weka_time*time_sensitivity)
            {
                System.out.println("PASS");
                passed++;
            }
            else
            {
                System.out.println("FAIL");
            }
            
        }catch(Exception ex)
        {
            System.out.println("FAIL");
            Logger.getLogger(test.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        System.out.println();
        
        try
        {
            System.out.print("Comparing to wekaPCA (no center, 1000x10000). weka takes: ");
            
            long weka_time = testSpeed(1000,10000,false,false);
            System.out.print(weka_time+" fast pca takes: ");
            
            
            long fpca_time = testSpeed(1000,10000,false,true);
            
            System.out.print(fpca_time+"\t");
            if(fpca_time < weka_time*time_sensitivity)
            {
                System.out.println("PASS");
                passed++;
            }
            else
            {
                System.out.println("FAIL");
            }
            
        }catch(Exception ex)
        {
            System.out.println("FAIL");
            Logger.getLogger(test.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        
        try
        {
            System.out.print("Comparing to wekaPCA (center, 1000x10000). weka takes: ");
            
            
            long weka_time = testSpeed(1000,10000,true,false);
            System.out.print(weka_time+" fast pca takes: ");
            
            
            long fpca_time = testSpeed(1000,10000,true,true);
            
            System.out.print(fpca_time+"\t");
            if(fpca_time < weka_time*time_sensitivity)
            {
                System.out.println("PASS");
                passed++;
            }
            else
            {
                System.out.println("FAIL");
            }
            
        }catch(Exception ex)
        {
            System.out.println("FAIL");
            Logger.getLogger(test.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        
    }
    
    /**
     * No tests, but just generate the execution time multiple times to test for significance
     */
    public static void calcTimeDistributions() throws Exception
    {
        /*Visual analysis can be done with the R code:
         * 
          density(data$execution_time[which(data$method == "weka_pca")]) -> weka
          density(data$execution_time[which(data$method == "fast_pca")]) -> fast
          plot(fast,col="red",xlab="Execution time (milisecs)")
          polygon(fast,col="red")
          polygon(weka,col="blue")
         * 
         */
        System.out.println("Multiple time tests:");
        System.out.println("method,execution_time");
        for(int ct=0;ct<100;ct++)
        {
            for(boolean fast : new boolean[]{true,false})
            {
                long time = testSpeed(100,10000,true,fast);
                System.out.println((fast?"fast_pca":"weka_pca")+","+time);
            }
        }
        System.out.println("\n\n");
    }
    public static void testAccuracy() throws Exception
    {
        //because the change between weka and faster_pca is relativly small, we don't require too many accuracy tests, just enough to catch any stupid mistakes
        //todo: add accuracy tests
        System.out.println("In accuracy tests");
        for(int ct=1;ct<=10;ct++)
        {
            System.out.print("Comparing centered results from PCA and faster_pca ("+ct+"/10)\tMean Sq Difference: ");
            RDG1 randomdatagen = new RDG1();
        randomdatagen.setNumAttributes(100);
        randomdatagen.setNumExamples(10000);
        randomdatagen.defineDataFormat();
        Instances data = randomdatagen.generateExamples();
        
        PrincipalComponents pca = new PrincipalComponents();
        faster_pca fpca = new faster_pca();
        
        pca.setCenterData(true);
        fpca.setCenterData(true);
        
        long time = System.currentTimeMillis();
        pca.setInputFormat(data);
        fpca.setInputFormat(data);
            
        Instances pca_r = Filter.useFilter(data, pca);
        Instances fpca_r = Filter.useFilter(data, fpca);
        
        double mean_sq_diff = 0;
        
        for(int i=0;i<pca_r.numInstances();i++)
        {
            Instance pca_r_i = pca_r.instance(i);
            Instance fpca_r_i = fpca_r.instance(i);
            
            for(int j=0;j< pca_r_i.numAttributes();j++)
            {
                mean_sq_diff += Math.pow(pca_r_i.value(j) - fpca_r_i.value(j), 2);
            }
        }
        
        System.out.print(mean_sq_diff/pca_r.numInstances()/pca_r.numAttributes()+"\t");
        
        if(mean_sq_diff < .001)
        {
            System.out.println("PASS");
            passed++;
        }
        else
        {
            System.out.println("FAIL");
        }
        
        }
        
        System.out.println();
        
        for(int ct=1;ct<=10;ct++)
        {
            System.out.print("Comparing non-centered results from PCA and faster_pca ("+ct+"/10)\tMean Sq Difference: ");
            RDG1 randomdatagen = new RDG1();
        randomdatagen.setNumAttributes(100);
        randomdatagen.setNumExamples(10000);
        randomdatagen.defineDataFormat();
        Instances data = randomdatagen.generateExamples();
        
        PrincipalComponents pca = new PrincipalComponents();
        faster_pca fpca = new faster_pca();
        
        pca.setCenterData(false);
        fpca.setCenterData(false);
        
        long time = System.currentTimeMillis();
        pca.setInputFormat(data);
        fpca.setInputFormat(data);
            
        Instances pca_r = Filter.useFilter(data, pca);
        Instances fpca_r = Filter.useFilter(data, fpca);
        
        double mean_sq_diff = 0;
        
        for(int i=0;i<pca_r.numInstances();i++)
        {
            Instance pca_r_i = pca_r.instance(i);
            Instance fpca_r_i = fpca_r.instance(i);
            
            for(int j=0;j< pca_r_i.numAttributes();j++)
            {
                mean_sq_diff += Math.pow(pca_r_i.value(j) - fpca_r_i.value(j), 2);
            }
        }
        
        System.out.print(mean_sq_diff/pca_r.numInstances()/pca_r.numAttributes()+"\t");
        
        if(mean_sq_diff < .001)
        {
            System.out.println("PASS");
            passed++;
        }
        else
        {
            System.out.println("FAIL");
        }
        
        }
    }
    
    public static void main(String args[]) throws Exception
    {
        calcTimeDistributions();
        
        
        System.out.println("Running tests of faster_pca\n");
        
        
        testAccuracy();
        System.out.println("\n\n");
        testSpeed();
        
        
        
        
        System.out.println("Passed "+passed+" / "+number_of_tests+" tests"+(number_of_tests==passed?"!":"."));
        
    }

   
}
