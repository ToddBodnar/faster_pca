
import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

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
        return super.input(instance);
    }
    
    public boolean batchFinished()
    {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    
}
