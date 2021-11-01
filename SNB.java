
package weka.classifiers.lsq;
 
import weka.core.*;

import static java.lang.Math.pow;

import java.util.Vector;
import weka.classifiers.*;

 
public class SNB extends AbstractClassifier{
	
	/** The number of class and each attribute value occurs in the data set*/
	private double [][] m_ClassAttCounts;
	/** The number of each class value occurs in the data set*/
	private double [] m_ClassCounts;
	
	/** The number of values for each attribute  in the data set*/
	private  int [] m_NumAttValues;
	
	/** The starting index of each attribute in the data set*/
	private int [] m_StartAttIndex;
	
	/** The number of  values for all attributes  in the data set*/
	private int m_TotalAttValues;
	
	/** The number  classes  in the data set*/
	private int m_NumClasses;
	
	/** The number of attributes including class in the data set*/
	private int m_NumAttributes;
	
	/** The number of instance in the data set*/
	private int m_NumInstances;
	
	/** The index of the class attribute in the data set*/
	private int m_ClassIndex;

	private Vector<Integer> vector;
	

	public void select(Instances instances) throws Exception {
		buildClassifier_(instances);
		vector = new Vector<Integer>();
		int m_NumClass = instances.numClasses();
        int[]count_class = new int[m_NumClass];
        int max_count=0;
        for(int i = 0;i<instances.numInstances();i++) {
        	count_class[(int) (instances.instance(i).classValue())]++;
        	max_count =Math.max(max_count, count_class[(int) (instances.instance(i).classValue())]);
        }
        double current_prob = (1.0 * max_count)/instances.numInstances();
        for(int i = 0; i < instances.numAttributes(); i++) {
        	int temp_index = 0;
        	double temp_prob=0;
        	for(int j = 0; j < instances.numAttributes(); j++) {
        		if(j == m_ClassIndex)	continue;
        		if(vector.contains(j) == true)	continue;
        		vector.addElement(j);
        		double prob = getCorrectClassified(instances);
        		 if(prob > temp_prob) {
        			 temp_prob = prob;
        			 temp_index = j;
        		 }
        		 vector.remove((Integer)j);
        	}
        	if(temp_prob >= current_prob) {
        		current_prob = temp_prob;
        		vector.addElement(temp_index);
        	}else break;
        }
	}
	public double getCorrectClassified(Instances instances) throws Exception {
		int cnt=0;
		int length = instances.numInstances();
		for(int i=0;i<length;i++) {
			double maxIndex= classifyInstance(instances.instance(i));
			if((int)maxIndex==(int)instances.instance(i).classValue()) {
				cnt++;
			}
		}
		
		return cnt*1.0/length;
	}
	public void buildClassifier_(Instances instances) {
		//reset variables
		m_NumClasses = instances.numClasses();
		m_ClassIndex = instances.classIndex();
		m_NumAttributes = instances.numAttributes();
		m_NumInstances = instances.numInstances();
		m_TotalAttValues = 0;
		// allocate space for attribute reference arrays
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];
		for(int i =0;i < m_NumAttributes;i++) {
			if(i != m_ClassIndex) {
				m_StartAttIndex[i]= m_TotalAttValues;
				m_NumAttValues[i]= instances.attribute(i).numValues();
				m_TotalAttValues +=m_NumAttValues[i];
			}
			else {
				m_StartAttIndex[i] = -1;
				m_NumAttValues[i] = m_NumClasses;
			}
		}
		//allocate space counts and frequencies
		m_ClassCounts = new double[m_NumClasses];
		m_ClassAttCounts = new double[m_NumClasses][10000];
		for(int k = 0;k<m_NumInstances;k++) {
			int classVal = (int)instances.instance(k).classValue();
			m_ClassCounts[classVal]++;
			int [] attIndex = new int [m_NumAttributes];
			for (int i = 0; i < m_NumAttributes; i++) {
				if(i == m_ClassIndex) {
					attIndex[i] = -1;
				}
				else {
					attIndex[i] = m_StartAttIndex[i]+(int)instances.instance(k).value(i);
					m_ClassAttCounts[classVal][attIndex[i]]++;
				}
				
			}
		}
	}
	public void buildClassifier(Instances instances)throws Exception{
		select(instances);
		buildClassifier_(instances);
	}
	
	public double[] distributionForInstance(Instance instance) throws Exception{
		//Definition of local variables
		double[] probs = new double[m_NumClasses];
		//store instance's attribute values in an int array
		int[] attIndex = new int[m_NumAttributes];
		for(int att = 0;att<m_NumAttributes;att++) {
			if(att==m_ClassIndex)
				attIndex[att]=-1;
			else {
				attIndex[att] = m_StartAttIndex[att]+(int)instance.value(att);
			}
		}
		double [][] weigh = new double[m_NumClasses][m_NumAttributes];
		double [] numAtt = new double[m_NumAttributes];
		for(int att = 0; att < m_NumAttributes; att++)
		{
		    numAtt[att] = 0;
		    if(attIndex[att] == -1) continue;
		    for(int classVal = 0; classVal < m_NumClasses; classVal++)
		    {
		        numAtt[att] += m_ClassAttCounts[classVal][attIndex[att]]; 
		    }
		    for(int classVal = 0; classVal < m_NumClasses; classVal++)
		    {
		        weigh[classVal][att] = (numAtt[att])/m_ClassAttCounts[classVal][attIndex[att]];
		    }
	     }

		for(int classVal =0;classVal<m_NumClasses;classVal++) {
			probs[classVal]=(m_ClassCounts[classVal]+1.0)/(m_NumInstances+m_NumClasses);
			for(int att_=0;att_<m_NumAttributes;att_++) {
				if(vector.contains(att_)==false)continue;
				if(attIndex[att_]==-1)continue;
				probs[classVal]*=pow((m_ClassAttCounts[classVal][attIndex[att_]] )/(m_ClassCounts[classVal]+m_NumAttValues[att_]),weigh[classVal][att_]);
			}
		}
		Utils.normalize(probs);
		return probs;
	}

    public static void main(String[] argv) {
		runClassifier(new SNB(), argv);
	}
 }


