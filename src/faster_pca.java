
import java.util.Enumeration;
import java.util.Vector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;
import weka.filters.unsupervised.attribute.Center;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * A -potentially- faster version of PCA than weka's PrincipalComponent filter
 * Code based on PrincipalComponents.java, Copyright (C) 2007-2012 University of Waikato, Hamilton, New Zealand by Mark HAll, Babi Schmidberger, and fracpete
 * @author toddbodnar
 */
public class faster_pca extends PrincipalComponents {

  /** for serialization. */
  private static final long serialVersionUID = -296639299L;

    @Override
    public String globalInfo() {
        return "PCA"; //todo: real info
    }



    public boolean input(Instance instance) throws java.lang.Exception
    {
       // System.out.println("test");
     return super.input(instance);
    
    }
    
    public boolean setInputFormat(Instances instances) throws Exception
    {
        return super.setInputFormat(instances);
    }
    
    /**
     * Modified version of PrincipalComponents.fillCovariance()
     * @throws Exception 
     */
    protected void fillCovariance() throws Exception {

    if (!super.getCenterData()) {
      fillCorrelation();
      return;
    }

    // now center the data by subtracting the mean
    m_centerFilter = new Center();
    m_centerFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances, m_centerFilter);

    // now compute the covariance matrix
    m_Correlation = new double[m_NumAttribs][m_NumAttribs];
    
    
    double trainInstancesCopy[][] = new double[m_NumInstances][m_NumAttribs];
    
    for(int i=0;i< m_NumInstances; i++)
    {
        Instance in = m_TrainInstances.instance(i);
        Enumeration enumer = in.enumerateAttributes();
        
        for(int j=0;j<m_NumAttribs;j++)
        {
            trainInstancesCopy[i][j] = in.value(j);
        }
    }

    for (int i = 0; i < m_NumAttribs; i++) {
      for (int j = 0; j < m_NumAttribs; j++) {

        double cov = 0;
        for (int k = 0; k < m_NumInstances; k++) {

          if (i == j) {
            //cov += (m_TrainInstances.instance(k).value(i) * m_TrainInstances.instance(k).value(i));
              cov += trainInstancesCopy[k][i] * trainInstancesCopy[k][i];
          } else {
            //cov += (m_TrainInstances.instance(k).value(i) * m_TrainInstances.instance(k).value(j));
              cov += trainInstancesCopy[k][i] * trainInstancesCopy[k][j];
          }
        }

        cov /= m_TrainInstances.numInstances() - 1;
        m_Correlation[i][j] = cov;
        m_Correlation[j][i] = cov;
      }
    }
  }
    
    
    /**
     * Modified version of PrincipalComponents.fillCorrelation()
     * @throws Exception 
     */
    protected void fillCorrelation() throws Exception {
    int i;
    int j;
    int k;
    double[] att1;
    double[] att2;
    double corr;

    m_Correlation = new double[m_NumAttribs][m_NumAttribs];
    att1 = new double[m_NumInstances];
    att2 = new double[m_NumInstances];
    
    double trainInstancesCopy[][] = new double[m_NumInstances][m_NumAttribs];
    
    for( i=0;i< m_NumInstances; i++)
    {
        Instance in = m_TrainInstances.instance(i);
        Enumeration enumer = in.enumerateAttributes();
        
        for( j=0;j<m_NumAttribs;j++)
        {
            trainInstancesCopy[i][j] = in.value(j);
        }
    }

    for (i = 0; i < m_NumAttribs; i++) {
      for (j = 0; j < m_NumAttribs; j++) {
        for (k = 0; k < m_NumInstances; k++) {
          //att1[k] = m_TrainInstances.instance(k).value(i);
          att1[k] = trainInstancesCopy[k][i];
          //att2[k] = m_TrainInstances.instance(k).value(j);
          att2[k] = trainInstancesCopy[k][j];
        }
        if (i == j) {
          m_Correlation[i][j] = 1.0;
        } else {
          corr = Utils.correlation(att1, att2, m_NumInstances);
          m_Correlation[i][j] = corr;
          m_Correlation[j][i] = corr;
        }
      }
    }

    // now standardize the input data
    m_standardizeFilter = new Standardize();
    m_standardizeFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances, m_standardizeFilter);
  }
    
    
    ///------ 1 to 1 copy from PrincipalComponents below
    /**
   * Initializes the filter with the given input data.
   * 
   * @param instances the data to process
   * @throws Exception in case the processing goes wrong
   * @see #batchFinished()
   */
  protected void setup(Instances instances) throws Exception {
    int i;
    int j;
    Vector<Integer> deleteCols;
    int[] todelete;
    double[][] v;
    Matrix corr;
    EigenvalueDecomposition eig;
    Matrix V;

    m_TrainInstances = new Instances(instances);

    // make a copy of the training data so that we can get the class
    // column to append to the transformed data (if necessary)
    m_TrainCopy = new Instances(m_TrainInstances, 0);

    m_ReplaceMissingFilter = new ReplaceMissingValues();
    m_ReplaceMissingFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances,
      m_ReplaceMissingFilter);

    m_NominalToBinaryFilter = new NominalToBinary();
    m_NominalToBinaryFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances,
      m_NominalToBinaryFilter);

    // delete any attributes with only one distinct value or are all missing
    deleteCols = new Vector<Integer>();
    for (i = 0; i < m_TrainInstances.numAttributes(); i++) {
      if (m_TrainInstances.numDistinctValues(i) <= 1) {
        deleteCols.addElement(i);
      }
    }

    if (m_TrainInstances.classIndex() >= 0) {
      // get rid of the class column
      m_HasClass = true;
      m_ClassIndex = m_TrainInstances.classIndex();
      deleteCols.addElement(new Integer(m_ClassIndex));
    }

    // remove columns from the data if necessary
    if (deleteCols.size() > 0) {
      m_AttributeFilter = new Remove();
      todelete = new int[deleteCols.size()];
      for (i = 0; i < deleteCols.size(); i++) {
        todelete[i] = (deleteCols.elementAt(i)).intValue();
      }
      m_AttributeFilter.setAttributeIndicesArray(todelete);
      m_AttributeFilter.setInvertSelection(false);
      m_AttributeFilter.setInputFormat(m_TrainInstances);
      m_TrainInstances = Filter.useFilter(m_TrainInstances, m_AttributeFilter);
    }

    // can evaluator handle the processed data ? e.g., enough attributes?
    getCapabilities().testWithFail(m_TrainInstances);

    m_NumInstances = m_TrainInstances.numInstances();
    m_NumAttribs = m_TrainInstances.numAttributes();

    // fillCorrelation();
    fillCovariance();

    // get eigen vectors/values
    corr = new Matrix(m_Correlation);
    eig = corr.eig();
    V = eig.getV();
    v = new double[m_NumAttribs][m_NumAttribs];
    for (i = 0; i < v.length; i++) {
      for (j = 0; j < v[0].length; j++) {
        v[i][j] = V.get(i, j);
      }
    }
    m_Eigenvectors = v.clone();
    m_Eigenvalues = eig.getRealEigenvalues().clone();

    // any eigenvalues less than 0 are not worth anything --- change to 0
    for (i = 0; i < m_Eigenvalues.length; i++) {
      if (m_Eigenvalues[i] < 0) {
        m_Eigenvalues[i] = 0.0;
      }
    }
    m_SortedEigens = Utils.sort(m_Eigenvalues);
    m_SumOfEigenValues = Utils.sum(m_Eigenvalues);

    m_TransformedFormat = determineOutputFormat(m_TrainInstances);
    setOutputFormat(m_TransformedFormat);

    m_TrainInstances = null;
  }
  
  /**
   * Signify that this batch of input to the filter is finished.
   * 
   * @return true if there are instances pending output
   * @throws NullPointerException if no input structure has been defined,
   * @throws Exception if there was a problem finishing the batch.
   */
  @Override
  public boolean batchFinished() throws Exception {
    int i;
    Instances insts;
    Instance inst;

    if (getInputFormat() == null) {
      throw new NullPointerException("No input instance format defined");
    }

    insts = getInputFormat();

    if (!isFirstBatchDone()) {
      setup(insts);
    }

    for (i = 0; i < insts.numInstances(); i++) {
      inst = convertInstance(insts.instance(i));
      inst.setDataset(getOutputFormat());
      push(inst);
    }

    flushInput();
    m_NewBatch = true;
    m_FirstBatchDone = true;

    return (numPendingOutput() != 0);
  }


    
}
