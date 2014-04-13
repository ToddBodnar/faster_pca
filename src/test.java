
import java.util.logging.Level;
import java.util.logging.Logger;
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
    public static final int number_of_tests = 2;
    public static int passed = 0;
    
    public static void testSpeed()
    {
        System.out.println("In speed tests");
        double time_sensitivity = .75; //how much lower we have to go to pass
        try
        {
            System.out.print("Comparing to wekaPCA (no center, 100x10000). weka takes: ");
            
            RDG1 randomdatagen = new RDG1();
            randomdatagen.setNumAttributes(100);
            randomdatagen.setNumExamples(10000);
            randomdatagen.defineDataFormat();
            Instances data = randomdatagen.generateExamples();
            
            PrincipalComponents pca = new PrincipalComponents();
            pca.setCenterData(false);
            
            long time = System.currentTimeMillis();
            pca.setInputFormat(data);
            
            Filter.useFilter(data, pca);
            long weka_time = System.currentTimeMillis() - time;
            
            System.out.print(weka_time+" fast pca takes: ");
            
            
            faster_pca fpca = new faster_pca();
            fpca.setCenterData(false);
            
            time = System.currentTimeMillis();
            fpca.setInputFormat(data);
            Filter.useFilter(data, fpca);
            long fpca_time = System.currentTimeMillis() - time;
            
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
            
            RDG1 randomdatagen = new RDG1();
            randomdatagen.setNumAttributes(100);
            randomdatagen.setNumExamples(10000);
            randomdatagen.defineDataFormat();
            Instances data = randomdatagen.generateExamples();
            
            PrincipalComponents pca = new PrincipalComponents();
            pca.setCenterData(true);
            
            long time = System.currentTimeMillis();
            pca.setInputFormat(data);
            
            Filter.useFilter(data, pca);
            long weka_time = System.currentTimeMillis() - time;
            
            System.out.print(weka_time+" fast pca takes: ");
            
            
            faster_pca fpca = new faster_pca();
            fpca.setCenterData(true);
            
            time = System.currentTimeMillis();
            fpca.setInputFormat(data);
            Filter.useFilter(data, fpca);
            long fpca_time = System.currentTimeMillis() - time;
            
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
    
    public static void main(String args[])
    {
        
        System.out.println("Running tests of faster_pca\n");
        
        testSpeed();
        
        
        
        
        System.out.println("Passed "+passed+" / "+number_of_tests+" tests"+(number_of_tests==passed?"!":"."));
        
    }
}
