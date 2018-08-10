package testCases;

import NeuralNetwork.NeuralNetwork;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;
import static org.junit.Assert.assertEquals;
import org.junit.Test;

/**
 * @author Ankit
 */
public class testFeedForward {

    public testFeedForward() {
    }
    
    @Test
    public void testFeedForward() {        
        ArrayList<double[]> target_list = new ArrayList<double[]>();
        ArrayList<double[]> input_list = new ArrayList<double[]>();
       
        String workingDir = System.getProperty("user.dir");        
        String csvFile = workingDir + "\\Arabic_Digits_csvTrainImages 60k x 784.csv";

        Scanner scanner = null;       
        String InputLine = "";        
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(csvFile)));
            scanner.hasNextLine();
            String ar = scanner.nextLine();
            while (scanner.hasNextLine()) { //start reading the file
                InputLine = scanner.nextLine();
                String[] InArray = InputLine.split(",");
                double[] input_array = new double[784];
                double[] target_array = new double[10];                
                //loop trhough entire record
                for (int i = 0; i < InArray.length; i++) {
                    //for the first column
                    if (i == 0) {   
                        int b = Integer.parseInt(InArray[i]);   //taken the label
                        for (int j = 0; j < 10; j++) {  //set of classification - upto 28                        
                            if (j == b) {   //for the input starting with 0
//                            if (j == (b-1)) {   //for the input starting with 1
                                target_array[j] = 1;    //put 1 for the corresponding digit
                            } else {
                                target_array[j] = 0;
                            }
                        }
                        target_list.add(target_array);  //added the target array
                    } else {    //put the remaining elements in the input array
                        input_array[i - 1] = Double.parseDouble(InArray[i]);
                    }
                }
                input_list.add(input_array); //added the input array
            } //end of while loop

        } catch (Exception e) {
            System.out.println("Error occured " + e);
        }
        
        NeuralNetwork test1 = new NeuralNetwork(784,392,10);
            
        for(int i = 0; i < input_list.size(); i++){
            test1.train(input_list.get(i), target_list.get(i), 1);
        }
        
        //start predicting !
        double[] predicted_outcome = test1.feedForward(input_list.get(0));
        double max = Integer.MIN_VALUE;
        int index = 0;
        //find the maximum of predicted outcome and that will be the actual prediction !
        for(int l = 0; l < predicted_outcome.length; l++){
            if(predicted_outcome[l] > max){
                max = predicted_outcome[l];
                index = l;
            }
        }
        assertEquals(0, index);          
    }
    
    @Test
    public void testFeedForward2() {
        ArrayList<double[]> target_list = new ArrayList<double[]>();
        ArrayList<double[]> input_list = new ArrayList<double[]>();
       
        String workingDir = System.getProperty("user.dir");        
        String csvFile = workingDir + "\\Arabic_Digits_csvTrainImages 60k x 784.csv";

        Scanner scanner = null;       
        String InputLine = "";        
        try {
            scanner = new Scanner(new BufferedReader(new FileReader(csvFile)));
            scanner.hasNextLine();
            String ar = scanner.nextLine();
            while (scanner.hasNextLine()) { //start reading the file
                InputLine = scanner.nextLine();
                String[] InArray = InputLine.split(",");
                double[] input_array = new double[784];
                double[] target_array = new double[10];                
                //loop trhough entire record
                for (int i = 0; i < InArray.length; i++) {
                    //for the first column
                    if (i == 0) {   
                        int b = Integer.parseInt(InArray[i]);   //taken the label
                        for (int j = 0; j < 10; j++) {  //set of classification - upto 28                        
                            if (j == b) {   //for the input starting with 0
//                            if (j == (b-1)) {   //for the input starting with 1
                                target_array[j] = 1;    //put 1 for the corresponding digit
                            } else {
                                target_array[j] = 0;
                            }
                        }
                        target_list.add(target_array);  //added the target array
                    } else {    //put the remaining elements in the input array
                        input_array[i - 1] = Double.parseDouble(InArray[i]);
                    }
                }
                input_list.add(input_array); //added the input array
            } //end of while loop

        } catch (Exception e) {
            System.out.println("Error occured " + e);
        }
        
        NeuralNetwork test1 = new NeuralNetwork(784,392,10);
            
        for(int i = 0; i < input_list.size(); i++){
            test1.train(input_list.get(i), target_list.get(i), 1);
        }
        
        //start predicting !
        double[] predicted_outcome = test1.feedForward(input_list.get(1));
        double max = Integer.MIN_VALUE;
        int index = 0;
        //find the maximum of predicted outcome and that will be the actual prediction !
        for(int l = 0; l < predicted_outcome.length; l++){
            if(predicted_outcome[l] > max){
                max = predicted_outcome[l];
                index = l;
            }
        }
        assertEquals(1, index);          
    }

}
